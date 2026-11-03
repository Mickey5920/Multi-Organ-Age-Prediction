from __future__ import annotations
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from .external_protocol import MIMIC_BARE_OMR_SENSITIVITY, MIMIC_PRIMARY_VARIANT, MIMIC_RX_SENSITIVITY, NHANES_PRIMARY_VARIANT, deterministic_hash_fold
MIMIC_VARIANTS = [MIMIC_PRIMARY_VARIANT, MIMIC_BARE_OMR_SENSITIVITY, MIMIC_RX_SENSITIVITY]
RELEASES = ['P_2017_MARCH2020_PREPANDEMIC', 'L_AUG2021_AUG2023']

def _canonical_icd(code: object, version: object) -> str:
    if code is None or pd.isna(code):
        raise ValueError('missing ICD code')
    normalized = unicodedata.normalize('NFKC', str(code))
    value = normalized.strip().upper().replace('.', '')
    if not value or any((character.isspace() for character in value)):
        raise ValueError('ICD code must not contain internal whitespace')
    if not value.isascii() or not value.isalnum():
        raise ValueError('invalid ICD code')
    version_value = int(version)
    if version_value == 10:
        if not value[0].isalpha() or not value[1:].isalnum():
            raise ValueError('invalid ICD10 code')
        return value
    if version_value == 9:
        return f'ICD9\x1f{value}'
    raise ValueError('unsupported ICD version')

def _participant_cohort(sources: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    patients = pd.read_csv(sources['mimic_patients'], compression='gzip', usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    admissions = pd.read_csv(sources['mimic_admissions'], compression='gzip', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime'])
    flow: dict[str, int] = {'raw_subjects': int(patients['subject_id'].nunique())}
    for column in ('admittime', 'dischtime', 'deathtime'):
        admissions[column] = pd.to_datetime(admissions[column], errors='coerce')
    demographics = patients.loc[patients[['gender', 'anchor_age', 'anchor_year']].notna().all(axis=1)].copy()
    flow['has_demographics'] = int(demographics['subject_id'].nunique())
    demographics = demographics.loc[demographics['gender'].isin(['M', 'F'])].copy()
    flow['sex_binary_m_or_f'] = int(demographics['subject_id'].nunique())
    demographics = demographics.loc[demographics['anchor_age'].ne(91)].copy()
    flow['anchor_age_not_91_topcoded'] = int(demographics['subject_id'].nunique())
    frame = admissions.merge(demographics[['subject_id', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='inner', validate='many_to_one')
    frame = frame.loc[frame['admittime'].notna() & frame['dischtime'].notna() & frame['dischtime'].gt(frame['admittime'])].copy()
    if not (frame['anchor_age'].eq(np.floor(frame['anchor_age'])) & frame['anchor_year'].eq(np.floor(frame['anchor_year']))).all():
        raise AssertionError('MIMIC anchor age/year are not integer resolution')
    frame['chronological_age'] = frame['anchor_age'].astype(int) + frame['dischtime'].dt.year.astype(int) - frame['anchor_year'].astype(int)
    frame['sex_male'] = frame['gender'].map({'M': 1, 'F': 0}).astype('int8')
    frame['hadm_numeric'] = pd.to_numeric(frame['hadm_id'], errors='raise')
    frame = frame.loc[frame['chronological_age'].ge(18)].copy()
    flow['adult_age_at_time_valid_discharge'] = int(frame['subject_id'].nunique())
    frame = frame.loc[frame['deathtime'].isna() | frame['deathtime'].gt(frame['dischtime'])].copy()
    flow['alive_at_time_valid_adult_discharge'] = int(frame['subject_id'].nunique())
    frame = frame.sort_values(['subject_id', 'admittime', 'dischtime', 'hadm_numeric'], kind='mergesort')
    frame['valid_admission_order'] = frame.groupby('subject_id', sort=False).cumcount() + 1
    has_two = frame.groupby('subject_id', sort=False)['valid_admission_order'].max().ge(2)
    flow['has_two_valid_adult_surviving_discharge_admissions'] = int(has_two.sum())
    index = frame.loc[frame['valid_admission_order'].eq(2)].copy()
    flow['second_valid_adult_surviving_discharge_selected'] = int(index['subject_id'].nunique())
    if index['subject_id'].duplicated().any():
        raise AssertionError('second valid admission is not unique')
    index['index_time'] = index['dischtime']
    index['hash_fold'] = index['subject_id'].map(deterministic_hash_fold).astype('int8')
    columns = ['subject_id', 'hadm_id', 'index_time', 'chronological_age', 'sex_male', 'hash_fold']
    return (index[columns].reset_index(drop=True), frame, flow)

def _mimic_diagnosis_events(cohort: pd.DataFrame, valid_admissions: pd.DataFrame, sources: Mapping[str, Path], preindex_diagnosis_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize only diagnoses from strictly pre-index valid admissions."""
    dictionary = pd.read_csv(sources['mimic_diagnosis_dictionary'], compression='gzip')
    dictionary['key'] = list(zip(dictionary['icd_code'].astype(str), dictionary['icd_version'].astype(int)))
    title = dict(zip(dictionary['key'], dictionary['long_title'].astype(str)))
    index = cohort.set_index('subject_id')['index_time']
    eligible = valid_admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']].copy()
    eligible['landmark_time'] = eligible['subject_id'].map(index)
    eligible = eligible.loc[eligible['landmark_time'].notna() & eligible['dischtime'].notna() & eligible['dischtime'].lt(eligible['landmark_time'])].copy()
    if eligible['hadm_id'].duplicated().any():
        raise AssertionError('MIMIC admission ID is not unique')
    admission_lookup = eligible.set_index('hadm_id')
    eligible_hadm = set(admission_lookup.index.tolist())
    rows: list[pd.DataFrame] = []
    invalid_rows: list[dict[str, Any]] = []
    preindex_only = pd.read_parquet(preindex_diagnosis_path)
    for raw_chunk in (preindex_only,):
        if not raw_chunk['hadm_id'].isin(eligible_hadm).all():
            raise AssertionError('pre-index materialization contains a forbidden admission')
        source_numbers = raw_chunk['source_row_number'].to_numpy(dtype=np.int64)
        chunk = raw_chunk.copy()
        if chunk.empty:
            continue
        chunk['source_row_number'] = source_numbers
        chunk['event_time'] = chunk['hadm_id'].map(admission_lookup['dischtime'])
        chunk['landmark_time'] = chunk['hadm_id'].map(admission_lookup['landmark_time'])
        canonical_codes: list[str | None] = []
        code_systems: list[str] = []
        valid_mask: list[bool] = []
        for record in chunk.itertuples(index=False):
            try:
                version = int(record.icd_version)
            except (TypeError, ValueError):
                version = -1
            code_system = 'ICD9' if version == 9 else 'ICD10' if version == 10 else 'not_applicable'
            try:
                canonical = _canonical_icd(record.icd_code, version)
            except (TypeError, ValueError):
                canonical = None
                invalid_rows.append({'modality': 'hospital_icd', 'code_system': code_system, 'reason': 'invalid_or_unsupported_code', 'participant_id': str(record.subject_id)})
            canonical_codes.append(canonical)
            code_systems.append(code_system)
            valid_mask.append(canonical is not None)
        chunk['canonical_code'] = canonical_codes
        chunk['code_system'] = code_systems
        chunk = chunk.loc[np.asarray(valid_mask, dtype=bool)].copy()
        if chunk.empty:
            continue
        chunk['canonical_description'] = [title.get((str(code), int(version)), f'ICD{int(version)} code {str(code)}') for code, version in zip(chunk['icd_code'], chunk['icd_version'])]
        chunk['participant_id'] = chunk['subject_id'].astype(str)
        chunk['source'] = 'hospital_icd'
        chunk['event_type_id'] = 1
        rows.append(chunk[['participant_id', 'event_time', 'landmark_time', 'source', 'code_system', 'event_type_id', 'canonical_code', 'canonical_description', 'source_row_number']])
    columns = ['participant_id', 'event_time', 'landmark_time', 'source', 'code_system', 'event_type_id', 'canonical_code', 'canonical_description', 'source_row_number']
    events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    audit = pd.DataFrame(invalid_rows, columns=['modality', 'code_system', 'reason', 'participant_id'])
    return (events, audit)

def _mimic_prescription_events(cohort: pd.DataFrame, sources: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = cohort.set_index('subject_id')['index_time']
    rows: list[pd.DataFrame] = []
    invalid_rows: list[dict[str, Any]] = []
    row_offset = 0
    for raw_chunk in pd.read_csv(sources['mimic_prescriptions'], compression='gzip', usecols=['subject_id', 'starttime', 'drug'], chunksize=1000000):
        source_numbers = np.arange(row_offset, row_offset + len(raw_chunk), dtype=np.int64)
        row_offset += len(raw_chunk)
        raw_chunk['landmark_time'] = raw_chunk['subject_id'].map(index)
        raw_chunk['event_time'] = pd.to_datetime(raw_chunk['starttime'], errors='coerce')
        preindex_mask = raw_chunk['landmark_time'].notna() & raw_chunk['event_time'].notna() & raw_chunk['event_time'].lt(raw_chunk['landmark_time'])
        chunk = raw_chunk.loc[preindex_mask].copy()
        if chunk.empty:
            continue
        chunk['source_row_number'] = source_numbers[preindex_mask.to_numpy()]
        chunk['canonical_description'] = chunk['drug'].map(lambda value: '' if pd.isna(value) else ' '.join(str(value).split()))
        invalid = chunk['canonical_description'].eq('')
        invalid_rows.extend(({'modality': 'inpatient_prescription_proxy', 'code_system': 'INPATIENT_RX', 'reason': 'invalid_or_missing_text', 'participant_id': str(subject_id)} for subject_id in chunk.loc[invalid, 'subject_id']))
        chunk = chunk.loc[~invalid].copy()
        if chunk.empty:
            continue
        chunk['participant_id'] = chunk['subject_id'].astype(str)
        chunk['source'] = 'inpatient_prescription_proxy'
        chunk['code_system'] = 'INPATIENT_RX'
        chunk['event_type_id'] = 2
        chunk['canonical_code'] = ''
        rows.append(chunk[['participant_id', 'event_time', 'landmark_time', 'source', 'code_system', 'event_type_id', 'canonical_code', 'canonical_description', 'source_row_number']])
    columns = ['participant_id', 'event_time', 'landmark_time', 'source', 'code_system', 'event_type_id', 'canonical_code', 'canonical_description', 'source_row_number']
    events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    audit = pd.DataFrame(invalid_rows, columns=['modality', 'code_system', 'reason', 'participant_id'])
    return (events, audit)

def _convert_formula(formula: str, value: pd.Series) -> pd.Series:
    formulas = {'value': 1.0, 'value * 2.54': 2.54, 'value * 0.45359237': 0.45359237, 'value * 88.4': 88.4, 'value * 10': 10.0, 'value * 17.104': 17.104, 'value * 0.357': 0.357, 'value * 0.2495': 0.2495, 'value * 0.02586': 0.02586, 'value * 0.0555': 0.0555, 'value * 0.3229': 0.3229, 'value * 0.0347': 0.0347, 'value * 0.01129': 0.01129, 'value * 59.48': 59.48, 'value * 2.496': 2.496}
    for prefix in ('value;', '51133:value; 52769:value/1000; 53157:value'):
        if formula.startswith(prefix):
            return value
    if formula == '(value - 2.15) * 10.929':
        return (value - 2.15) * 10.929
    if formula not in formulas:
        raise ValueError(f'unsupported frozen formula: {formula}')
    return value * formulas[formula]

def _value_rule_accepts(value: float, rule: str, source_unit: str) -> bool:
    if not np.isfinite(value):
        return False
    if '2<=value<=25' in rule:
        return 2.0 <= value <= 25.0
    if 'percent branch 0<=value<=100' in rule:
        return 0.0 <= value <= 100.0 if source_unit == '%' else value >= 0.0
    if '0<=value<=100' in rule or source_unit == '%':
        return 0.0 <= value <= 100.0
    if 'negative values rejected' in rule:
        return value >= 0.0
    raise ValueError(f'unsupported frozen primary_value_rule: {rule}')

def _exact_omr_selector_names(selector: str) -> set[str]:
    if selector.startswith("result_name == '") and selector.endswith("'"):
        return {selector[len("result_name == '"):-1]}
    prefix = 'result_name in {'
    if selector.startswith(prefix) and selector.endswith('}'):
        return {name for name in selector[len(prefix):-1].split(',') if name}
    raise ValueError(f'unsupported frozen OMR selector: {selector}')

def _lab_available_preindex_mask(frame: pd.DataFrame) -> pd.Series:
    return frame['charttime'].notna() & frame['storetime'].notna() & frame['valuenum'].notna() & frame['charttime'].le(frame['index_time']) & frame['storetime'].le(frame['index_time']) & frame['charttime'].ge(frame['index_time'] - pd.Timedelta(days=365))

def _omr_available_preindex_mask(frame: pd.DataFrame) -> pd.Series:
    index_date = frame['index_time'].dt.normalize()
    return frame['chartdate'].notna() & frame['chartdate'].lt(index_date) & frame['chartdate'].ge(index_date - pd.Timedelta(days=365))

def _build_mimic_features(cohort: pd.DataFrame, artifact_dir: Path, sources: Mapping[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    mapping = pd.read_csv(artifact_dir / 'MIMIC_78_SLOT_MAP.csv', keep_default_na=False)
    subject_order = cohort['subject_id'].astype(int).tolist()
    subject_to_row = {subject_id: index for index, subject_id in enumerate(subject_order)}
    index_time = cohort.set_index('subject_id')['index_time']
    raw = np.zeros((len(cohort), 78), dtype=np.float32)
    missing = np.ones((len(cohort), 78), dtype=bool)
    coverage = []
    lab_map: dict[tuple[int, str], list[tuple[int, str, str]]] = {}
    for row in mapping.loc[mapping['source_file_id'].eq('mimic_labevents')].itertuples(index=False):
        for branch in row.primary_source_selector.split('|'):
            itemid = int(branch.split('[', 1)[0])
            unit = branch.split('valueuom=', 1)[1].rstrip(']')
            lab_map.setdefault((itemid, unit), []).append((int(row.slot_index), row.primary_formula, row.primary_value_rule))
    latest: dict[tuple[int, int], tuple[pd.Timestamp, pd.Timestamp, int, float]] = {}
    bare_extra: dict[tuple[int, int], tuple[pd.Timestamp, pd.Timestamp, int, float]] = {}
    allowed_itemids = {key[0] for key in lab_map}
    for chunk in pd.read_csv(sources['mimic_labevents'], compression='gzip', usecols=['labevent_id', 'subject_id', 'itemid', 'charttime', 'storetime', 'valuenum', 'valueuom'], chunksize=1000000):
        chunk = chunk.loc[chunk['subject_id'].isin(subject_to_row) & chunk['itemid'].isin(allowed_itemids)].copy()
        if chunk.empty:
            continue
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk['storetime'] = pd.to_datetime(chunk['storetime'], errors='coerce')
        chunk['index_time'] = chunk['subject_id'].map(index_time)
        chunk = chunk.loc[_lab_available_preindex_mask(chunk)]
        for record in chunk.itertuples(index=False):
            branches = lab_map.get((int(record.itemid), str(record.valueuom)), [])
            for slot, formula, value_rule in branches:
                value = float(record.valuenum)
                if not _value_rule_accepts(value, value_rule, str(record.valueuom)):
                    continue
                if slot == 24 and int(record.itemid) == 52769:
                    converted = value / 1000.0
                else:
                    converted = float(_convert_formula(formula, pd.Series([value])).iloc[0])
                key = (int(record.subject_id), slot)
                candidate = (record.charttime, record.storetime, int(record.labevent_id), converted)
                if key not in latest or candidate[:3] > latest[key][:3]:
                    latest[key] = candidate
    omr_rows = mapping.loc[mapping['source_file_id'].eq('mimic_omr')]
    allowed_names = {'Height', 'Weight'}
    for row in omr_rows.itertuples(index=False):
        allowed_names.update(_exact_omr_selector_names(row.primary_source_selector))
    for chunk in pd.read_csv(sources['mimic_omr'], compression='gzip', usecols=['subject_id', 'chartdate', 'seq_num', 'result_name', 'result_value'], chunksize=500000):
        chunk = chunk.loc[chunk['subject_id'].isin(subject_to_row) & chunk['result_name'].isin(allowed_names)].copy()
        chunk['chartdate'] = pd.to_datetime(chunk['chartdate'], errors='coerce')
        chunk['index_time'] = chunk['subject_id'].map(index_time)
        chunk = chunk.loc[_omr_available_preindex_mask(chunk)]
        for record in chunk.itertuples(index=False):
            name = str(record.result_name)
            text = str(record.result_value)
            if name in {'Height', 'Weight'}:
                try:
                    source_value = float(text)
                except ValueError:
                    source_value = math.nan
                slot = 3 if name == 'Height' else 11
                converted = math.nan
                if name == 'Height' and 40 <= source_value <= 90:
                    converted = source_value * 2.54
                elif name == 'Height' and 120 <= source_value <= 230:
                    converted = source_value
                elif name == 'Weight' and 50 <= source_value <= 1000:
                    converted = source_value * 0.45359237
                if np.isfinite(converted):
                    key = (int(record.subject_id), slot)
                    candidate = (record.chartdate, record.chartdate, int(record.seq_num), float(converted))
                    if key not in bare_extra or candidate[:3] > bare_extra[key][:3]:
                        bare_extra[key] = candidate
            for row in omr_rows.itertuples(index=False):
                if name not in _exact_omr_selector_names(row.primary_source_selector):
                    continue
                if int(row.slot_index) in {8, 9}:
                    parts = text.split('/')
                    if len(parts) != 2:
                        continue
                    try:
                        systolic, diastolic = map(float, parts)
                    except ValueError:
                        continue
                    if not (20 <= diastolic <= 200 and 40 <= systolic <= 300 and (systolic > diastolic)):
                        continue
                    value = diastolic if int(row.slot_index) == 8 else systolic
                else:
                    try:
                        source_value = float(text)
                    except ValueError:
                        continue
                    slot = int(row.slot_index)
                    if slot == 3 and (not 40 <= source_value <= 90):
                        continue
                    if slot == 10 and (not 10 <= source_value <= 100):
                        continue
                    if slot == 11 and (not 50 <= source_value <= 1000):
                        continue
                    value = float(_convert_formula(row.primary_formula, pd.Series([source_value])).iloc[0])
                key = (int(record.subject_id), int(row.slot_index))
                candidate = (record.chartdate, record.chartdate, int(record.seq_num), value)
                if key not in latest or candidate[:3] > latest[key][:3]:
                    latest[key] = candidate
    for (subject_id, slot), (*_, value) in latest.items():
        row_index = subject_to_row[subject_id]
        raw[row_index, slot] = value
        missing[row_index, slot] = False
    raw_bare = raw.copy()
    missing_bare = missing.copy()
    for key, candidate in bare_extra.items():
        primary_candidate = latest.get(key)
        if primary_candidate is not None and primary_candidate[:3] >= candidate[:3]:
            continue
        subject_id, slot = key
        row_index = subject_to_row[subject_id]
        raw_bare[row_index, slot] = candidate[3]
        missing_bare[row_index, slot] = False
    for row in mapping.itertuples(index=False):
        slot = int(row.slot_index)
        for variant in MIMIC_VARIANTS:
            is_bare = variant == MIMIC_BARE_OMR_SENSITIVITY
            variant_missing = missing_bare if is_bare else missing
            observed = int((~variant_missing[:, slot]).sum())
            coverage.append({'input_variant': variant, 'slot_index': slot, 'feature_name': row.feature_name, 'map_status': row.map_status, 'observed_n': observed, 'eligible_n': len(cohort), 'observed_fraction': observed / len(cohort) if len(cohort) else math.nan, 'source_selector': row.sensitivity_source_selector if is_bare else row.primary_source_selector, 'source_unit': row.sensitivity_source_unit if is_bare else row.primary_source_unit, 'target_unit': row.target_unit, 'formula': row.sensitivity_formula if is_bare else row.primary_formula, 'value_rule': row.sensitivity_value_rule if is_bare else row.primary_value_rule, 'all_78_slots_adjudicated': True})
    return (raw, missing, raw_bare, missing_bare, pd.DataFrame(coverage))

def _build_nhanes_features(artifact_dir: Path, sources: Mapping[str, Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    mapping = pd.read_csv(artifact_dir / 'NHANES_78_SLOT_RELEASE_MAP.csv', keep_default_na=False)
    participants = []
    raw_parts = []
    missing_parts = []
    coverage_rows = []
    for release_id, demo_id in zip(RELEASES, ['nhanes_p_demo', 'nhanes_l_demo']):
        demo = pd.read_sas(sources[demo_id], format='xport', encoding='latin1')
        primary_age_candidate = _nhanes_primary_integer_age_guard(demo['RIDAGEYR'], demo['RIDSTATR'])
        demo = demo.loc[primary_age_candidate & demo['RIAGENDR'].isin([1, 2])].copy()
        demo['participant_id'] = demo['SEQN'].astype(int).astype(str)
        demo['chronological_age'] = demo['RIDAGEYR'].astype(float)
        demo['sex_male'] = demo['RIAGENDR'].map({1: 1, 2: 0}).astype('int8')
        demo['release_id'] = release_id
        demo['hash_fold'] = demo['participant_id'].map(deterministic_hash_fold).astype('int8')
        frames: dict[str, pd.DataFrame] = {}
        release_map = mapping.loc[mapping['release_id'].eq(release_id)]
        values = np.zeros((len(demo), 78), dtype=np.float32)
        masks = np.ones((len(demo), 78), dtype=bool)
        seqn_order = demo['SEQN'].to_numpy()
        for row in release_map.itertuples(index=False):
            if not row.source_file_id:
                coverage_rows.append({'release_id': release_id, 'input_variant': NHANES_PRIMARY_VARIANT, 'slot_index': int(row.slot_index), 'feature_name': row.feature_name, 'map_status': row.map_status, 'observed_n': 0, 'eligible_n': len(demo), 'observed_fraction': 0.0, 'all_78_slots_adjudicated': True})
                continue
            if row.source_file_id not in frames:
                frames[row.source_file_id] = pd.read_sas(sources[row.source_file_id], format='xport', encoding='latin1').set_index('SEQN')
            source = frames[row.source_file_id]
            variables = row.source_variables.split('|')
            aligned = source.reindex(seqn_order)
            if row.formula in variables:
                result = pd.to_numeric(aligned[row.formula], errors='coerce')
            elif row.formula == 'URXUCR * 88.4':
                result = pd.to_numeric(aligned['URXUCR'], errors='coerce') * 88.4
            elif row.formula == 'LBXPLTSI * LBXMPSI / 10000.0':
                result = pd.to_numeric(aligned['LBXPLTSI'], errors='coerce') * pd.to_numeric(aligned['LBXMPSI'], errors='coerce') / 10000.0
            elif row.formula == '(LBXGH - 2.15) * 10.929':
                result = (pd.to_numeric(aligned['LBXGH'], errors='coerce') - 2.15) * 10.929
            elif row.formula.startswith('BMXWAIST / BMXHIP'):
                waist = pd.to_numeric(aligned['BMXWAIST'], errors='coerce')
                hip = pd.to_numeric(aligned['BMXHIP'], errors='coerce')
                result = waist / hip.where(hip > 0)
            else:
                raise ValueError(f'unsupported NHANES frozen formula: {row.formula}')
            observed = result.notna().to_numpy()
            values[observed, int(row.slot_index)] = result.loc[observed].to_numpy(dtype=np.float32)
            masks[observed, int(row.slot_index)] = False
            coverage_rows.append({'release_id': release_id, 'input_variant': NHANES_PRIMARY_VARIANT, 'slot_index': int(row.slot_index), 'feature_name': row.feature_name, 'map_status': row.map_status, 'observed_n': int(observed.sum()), 'eligible_n': len(demo), 'observed_fraction': float(observed.mean()) if len(observed) else math.nan, 'all_78_slots_adjudicated': True})
        participants.append(demo[['participant_id', 'release_id', 'chronological_age', 'sex_male', 'hash_fold', 'SEQN', 'SDMVSTRA', 'SDMVPSU', 'WTMECPRP' if release_id == RELEASES[0] else 'WTMEC2YR']])
        raw_parts.append(values)
        missing_parts.append(masks)
    return (pd.concat(participants, ignore_index=True), np.concatenate(raw_parts), np.concatenate(missing_parts), pd.DataFrame(coverage_rows))

def _nhanes_primary_integer_age_guard(age: pd.Series, ridstatr: pd.Series) -> pd.Series:
    values = pd.to_numeric(age, errors='coerce')
    candidate = ridstatr.eq(2) & values.between(40, 73, inclusive='both')
    if (candidate & values.ne(np.floor(values))).any():
        raise AssertionError('NHANES primary age contains fractional RIDAGEYR')
    return candidate

def _build_nhanes_survey_design_universe(sources: Mapping[str, Path]) -> pd.DataFrame:
    """Retain the full examined release design; 40-73 is a domain flag only."""
    rows = []
    for release_id, demo_id, weight_variable in ((RELEASES[0], 'nhanes_p_demo', 'WTMECPRP'), (RELEASES[1], 'nhanes_l_demo', 'WTMEC2YR')):
        demo = pd.read_sas(sources[demo_id], format='xport', encoding='latin1')
        required = {'SEQN', 'RIDSTATR', 'RIDAGEYR', 'RIAGENDR', 'SDMVSTRA', 'SDMVPSU', weight_variable}
        if not required.issubset(demo.columns):
            raise AssertionError(f'NHANES full design columns missing: {release_id}')
        demo = demo.loc[demo['RIDSTATR'].eq(2)].copy()
        age = pd.to_numeric(demo['RIDAGEYR'], errors='coerce')
        domain = age.between(40, 73, inclusive='both') & age.eq(np.floor(age)) & demo['RIAGENDR'].isin([1, 2])
        frame = pd.DataFrame({'participant_id': demo['SEQN'].astype(int).astype(str), 'release_id': release_id, 'chronological_age': age.astype(float), 'sex_male': demo['RIAGENDR'].map({1: 1, 2: 0}), 'SDMVSTRA': demo['SDMVSTRA'], 'SDMVPSU': demo['SDMVPSU'], 'survey_weight': pd.to_numeric(demo[weight_variable], errors='coerce'), 'weight_variable': weight_variable, 'domain_40_73': domain.astype(bool)})
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(['release_id', 'participant_id']).any():
        raise AssertionError('NHANES full survey design key is not unique')
    return result

def _pack_events(events: pd.DataFrame, participant_order: Sequence[str], code_to_id: Mapping[str, int], text_to_id: Mapping[tuple[str, str], int]) -> dict[str, np.ndarray]:
    n = len(participant_order)
    packed = {'event_code_id': np.zeros((n, 32), dtype=np.int64), 'event_text_id': np.zeros((n, 32), dtype=np.int64), 'event_relative_days': np.zeros((n, 32), dtype=np.int32), 'event_type_id': np.zeros((n, 32), dtype=np.int8), 'event_valid_mask': np.zeros((n, 32), dtype=bool)}
    grouped = {str(key): value for key, value in events.groupby('participant_id', sort=False)}
    for row_index, participant_id in enumerate(participant_order):
        group = grouped.get(str(participant_id))
        if group is None:
            continue
        if len(group) > 32:
            raise AssertionError('packed group exceeds L32')
        for column_index, record in enumerate(group.itertuples(index=False)):
            is_icd = record.source == 'hospital_icd'
            role = 'icd_description' if is_icd else 'inpatient_prescription_proxy'
            packed['event_code_id'][row_index, column_index] = code_to_id[str(record.canonical_code)] if is_icd else 0
            packed['event_text_id'][row_index, column_index] = text_to_id[role, str(record.canonical_description)]
            relative = math.floor((pd.Timestamp(record.event_time) - pd.Timestamp(record.landmark_time)).total_seconds() / 86400.0)
            if relative >= 0:
                raise AssertionError('packed event is not strictly pre-index')
            packed['event_relative_days'][row_index, column_index] = relative
            packed['event_type_id'][row_index, column_index] = int(record.event_type_id)
            packed['event_valid_mask'][row_index, column_index] = True
    return packed

def _icd10_organs(code: str, mapping: pd.DataFrame) -> set[str]:
    try:
        canonical = _canonical_icd(code, 10)
    except ValueError:
        return set()
    category = canonical[:3]
    result = set()
    for row in mapping.itertuples(index=False):
        if str(row.category_start) <= category <= str(row.category_end):
            result.add(str(row.organ_system))
    return result

def _build_nhanes_questionnaire_labels(artifact_dir: Path, sources: Mapping[str, Path], participants: pd.DataFrame) -> pd.DataFrame:
    mapping = pd.read_csv(artifact_dir / 'NHANES_QUESTIONNAIRE_LABEL_MAP.csv', keep_default_na=False)
    output = []
    loaded: dict[str, pd.DataFrame] = {}
    for row in mapping.itertuples(index=False):
        release = str(row.release_id)
        subset = participants.loc[participants['release_id'].eq(release), ['participant_id', 'SEQN']].copy()
        if row.estimability == 'not_estimable':
            for participant_id in subset['participant_id']:
                output.append({'participant_id': str(participant_id), 'release_id': release, 'organ_system': row.organ_system, 'label': np.nan, 'label_status': 'not_estimable_by_design'})
            continue
        merged = subset.set_index('SEQN')
        variables = str(row.source_variables).split('|')
        values = []
        for source_id in str(row.source_file_ids).split('|'):
            if source_id not in loaded:
                loaded[source_id] = pd.read_sas(sources[source_id], format='xport', encoding='latin1').set_index('SEQN')
            available = [variable for variable in variables if variable in loaded[source_id].columns]
            if available:
                values.append(loaded[source_id][available].reindex(merged.index))
        matrix = pd.concat(values, axis=1).loc[:, ~pd.concat(values, axis=1).columns.duplicated()]
        labels = _questionnaire_composite_labels(matrix)
        for participant_id, label in zip(merged['participant_id'], labels):
            output.append({'participant_id': str(participant_id), 'release_id': release, 'organ_system': row.organ_system, 'label': label, 'label_status': 'estimable' if np.isfinite(label) else 'missing_questionnaire_response'})
    return pd.DataFrame(output)

def _questionnaire_composite_labels(matrix: pd.DataFrame) -> np.ndarray:
    positive = matrix.eq(1).any(axis=1)
    negative = matrix.notna().all(axis=1) & matrix.eq(2).all(axis=1)
    return np.where(positive, 1.0, np.where(negative, 0.0, np.nan))

def _completed_diagnosis_endpoint_history(diagnosis: pd.DataFrame, valid_admissions: pd.DataFrame, cohort: pd.DataFrame, icd10_map: pd.DataFrame) -> tuple[set[int], set[tuple[int, str]], pd.Series]:
    value = diagnosis.merge(valid_admissions[['subject_id', 'hadm_id', 'dischtime']], on=['subject_id', 'hadm_id'], how='left', validate='many_to_one')
    index = cohort.set_index('subject_id')['index_time']
    value['index_time'] = value['subject_id'].map(index)
    value = value.loc[value['index_time'].notna() & value['dischtime'].notna()].copy()
    excluded_icd9 = set(map(int, value.loc[value['dischtime'].le(value['index_time']) & value['icd_version'].eq(9), 'subject_id'].unique()))
    value = value.loc[value['icd_version'].eq(10)].copy()
    value['organs'] = value['icd_code'].map(lambda code: _icd10_organs(str(code), icd10_map))
    value = value.loc[value['organs'].map(bool)]
    expanded = []
    for row in value.itertuples(index=False):
        for organ in sorted(row.organs):
            expanded.append({'subject_id': int(row.subject_id), 'hadm_id': int(row.hadm_id), 'diagnosis_available_time': row.dischtime, 'seq_num': int(row.seq_num), 'organ_system': organ, 'pre_or_index': row.dischtime <= row.index_time})
    organ_diagnosis = pd.DataFrame(expanded, columns=['subject_id', 'hadm_id', 'diagnosis_available_time', 'seq_num', 'organ_system', 'pre_or_index'])
    if organ_diagnosis.empty:
        return (excluded_icd9, set(), pd.Series(dtype='datetime64[ns]'))
    baseline = set(zip(organ_diagnosis.loc[organ_diagnosis['pre_or_index'], 'subject_id'], organ_diagnosis.loc[organ_diagnosis['pre_or_index'], 'organ_system']))
    post = organ_diagnosis.loc[~organ_diagnosis['pre_or_index']].sort_values(['subject_id', 'organ_system', 'diagnosis_available_time', 'seq_num', 'hadm_id'], kind='mergesort')
    first_post = post.drop_duplicates(['subject_id', 'organ_system'], keep='first').set_index(['subject_id', 'organ_system'])['diagnosis_available_time']
    return (excluded_icd9, baseline, first_post)

def _date_only_dod_rule(index_time: pd.Timestamp, death_date: pd.Timestamp | None) -> dict[str, Any]:
    index_date = pd.Timestamp(index_time).normalize()
    if death_date is None or pd.isna(death_date):
        return {'status': 'no_recorded_dod', 'normalized_death': pd.NaT, 'death_days': None, 'same_day_death': False}
    normalized_death = pd.Timestamp(death_date).normalize()
    if normalized_death < index_date:
        return {'status': 'excluded_inconsistent_dod_before_index_date', 'normalized_death': normalized_death, 'death_days': None, 'same_day_death': False}
    same_day = normalized_death == index_date
    return {'status': 'endpoint_eligible', 'normalized_death': normalized_death, 'death_days': float((normalized_death - index_date).days), 'same_day_death': bool(same_day)}


def _materialize_preindex_diagnosis_allowlist(
    cohort: pd.DataFrame,
    valid_admissions: pd.DataFrame,
    diagnosis_source: Path,
    destination: Path,
) -> Path:
    """Arrow-filter raw diagnosis batches before any conversion to pandas."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    index = cohort.set_index("subject_id")["index_time"]
    eligible = valid_admissions[
        ["subject_id", "hadm_id", "admittime", "dischtime"]
    ].copy()
    eligible["landmark_time"] = eligible["subject_id"].map(index)
    eligible = eligible.loc[
        eligible["landmark_time"].notna()
        & eligible["dischtime"].notna()
        & eligible["dischtime"].lt(eligible["landmark_time"])
    ].copy()
    if eligible["hadm_id"].duplicated().any():
        raise AssertionError("MIMIC pre-index admission allowlist is not unique")
    eligible_hadm = np.sort(
        pd.to_numeric(eligible["hadm_id"], errors="raise").to_numpy(dtype=np.int64)
    )
    allowed = pa.array(eligible_hadm, type=pa.int64())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    stream = pa.input_stream(str(diagnosis_source), compression="gzip")
    reader = pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(block_size=8 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(
            include_columns=[
                "subject_id",
                "hadm_id",
                "seq_num",
                "icd_code",
                "icd_version",
            ],
            column_types={"subject_id": pa.int64(), "hadm_id": pa.int64()},
        ),
    )
    writer: pq.ParquetWriter | None = None
    raw_rows = 0
    selected_rows = 0
    raw_batches = 0
    try:
        for batch in reader:
            raw_batches += 1
            batch_rows = batch.num_rows
            mask = pc.is_in(batch.column("hadm_id"), value_set=allowed)
            selected_index = pc.indices_nonzero(mask)
            filtered = batch.take(selected_index)
            global_row_number = pc.add(
                pc.cast(selected_index, pa.int64()), pa.scalar(raw_rows, pa.int64())
            )
            raw_rows += batch_rows
            if filtered.num_rows == 0:
                continue
            filtered = pa.RecordBatch.from_arrays(
                [*filtered.columns, global_row_number],
                names=[*filtered.schema.names, "source_row_number"],
            )
            if not pc.all(pc.is_in(filtered.column("hadm_id"), value_set=allowed)).as_py():
                raise AssertionError("post-index admission escaped Arrow predicate")
            if writer is None:
                writer = pq.ParquetWriter(destination, filtered.schema)
            writer.write_batch(filtered)
            selected_rows += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
        stream.close()
    if writer is None:
        empty = pa.table(
            {
                "subject_id": pa.array([], type=pa.int64()),
                "hadm_id": pa.array([], type=pa.int64()),
                "seq_num": pa.array([], type=pa.float64()),
                "icd_code": pa.array([], type=pa.string()),
                "icd_version": pa.array([], type=pa.float64()),
                "source_row_number": pa.array([], type=pa.int64()),
            }
        )
        pq.write_table(empty, destination)
    return destination
