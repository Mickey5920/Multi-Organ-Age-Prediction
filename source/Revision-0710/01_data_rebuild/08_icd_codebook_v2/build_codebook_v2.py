from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_V1 = (
    ROOT
    / "Revision-0710"
    / "01_data_rebuild"
    / "06_icd_codebook_v1"
    / "02_v1_corrected_primary"
    / "mapping.csv"
)
DEFAULT_OBSERVED = (
    ROOT
    / "Revision-0710"
    / "01_data_rebuild"
    / "01_index_and_events"
    / "observed_icd_code_system_matches.csv"
)
DEFAULT_COHORT = (
    ROOT
    / "Revision-0710"
    / "01_data_rebuild"
    / "01_index_and_events"
    / "age_model_cohort.parquet"
)
DEFAULT_RAW = Path(
    "E:/MPU/Project/Dataset-0911/Process/data/"
    "1-combined_ICD_Time_birth_phenotype_age_all-1124.csv"
)

ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)


@dataclass(frozen=True)
class Rule:
    icd_prefix: str
    code_title: str
    primary_organ: str
    secondary_organ: str
    baseline_chronic_eligible: bool
    incident_endpoint_eligible: bool
    recurrent_endpoint_eligible: bool
    acute_or_chronic: str
    malignancy_flag: bool
    inclusion_rationale: str
    review_status: str = "reviewed_against_who_icd10_and_ukb_coding19"


def family(letter: str, number: int) -> str:
    return f"{letter}{number:02d}"


def normalize_code(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper()).strip()


def add_rule(
    rules: dict[str, Rule],
    prefix: str,
    title: str,
    primary: str,
    *,
    secondary: str = "",
    baseline: bool,
    incident: bool,
    recurrent: bool | None = None,
    course: str,
    malignancy: bool = False,
    rationale: str,
    review_status: str = "reviewed_against_who_icd10_and_ukb_coding19",
) -> None:
    prefix = normalize_code(prefix)
    if recurrent is None:
        recurrent = incident
    if primary and primary not in ORGANS:
        raise ValueError(f"Unknown primary organ {primary!r} for {prefix}")
    secondary_organs = [item for item in secondary.split(";") if item]
    if any(item not in ORGANS for item in secondary_organs):
        raise ValueError(f"Unknown secondary organ {secondary!r} for {prefix}")
    if primary and primary in secondary_organs:
        raise ValueError(f"Primary organ repeated as secondary for {prefix}")
    rules[prefix] = Rule(
        icd_prefix=prefix,
        code_title=title,
        primary_organ=primary,
        secondary_organ=";".join(secondary_organs),
        baseline_chronic_eligible=baseline,
        incident_endpoint_eligible=incident,
        recurrent_endpoint_eligible=bool(recurrent),
        acute_or_chronic=course,
        malignancy_flag=malignancy,
        inclusion_rationale=rationale,
        review_status=review_status,
    )


def add_families(
    rules: dict[str, Rule],
    prefixes: list[str],
    title: str,
    primary: str,
    **kwargs: object,
) -> None:
    for prefix in prefixes:
        add_rule(rules, prefix, title, primary, **kwargs)


def build_rules() -> dict[str, Rule]:
    rules: dict[str, Rule] = {}

    # Cardiovascular. Acute myocardial infarction, stroke and embolic disease
    # remain eligible because a prior event represents established organ burden.
    add_families(
        rules,
        [family("I", value) for value in range(0, 3)],
        "Acute rheumatic fever",
        "cardiovascular",
        baseline=False,
        incident=False,
        course="acute",
        rationale="Acute rheumatic fever is retained only in broad morbidity sensitivity.",
    )
    for start, end, title in (
        (5, 9, "Chronic rheumatic heart disease"),
        (10, 15, "Hypertensive disease"),
        (20, 25, "Ischaemic heart disease"),
        (26, 28, "Pulmonary heart and pulmonary-circulation disease"),
        (31, 52, "Other established heart disease"),
        (60, 69, "Cerebrovascular disease"),
        (70, 79, "Arterial disease"),
        (80, 89, "Venous and lymphatic vascular disease"),
    ):
        add_families(
            rules,
            [family("I", value) for value in range(start, end + 1)],
            title,
            "cardiovascular",
            baseline=True,
            incident=True,
            course="chronic_or_established_event",
            rationale="WHO ICD-10 circulatory-system disease with persistent burden or a clinically established event.",
        )
    add_families(
        rules,
        [family("I", value) for value in range(21, 25)],
        "Acute ischaemic cardiovascular event",
        "cardiovascular",
        baseline=True,
        incident=True,
        course="acute_event_with_persistent_history",
        rationale="Myocardial infarction and acute ischaemic events are valid incident endpoints and establish prevalent cardiovascular disease thereafter.",
    )
    add_rule(
        rules,
        "I30",
        "Acute pericarditis",
        "cardiovascular",
        baseline=False,
        incident=True,
        course="acute",
        rationale="Eligible as an incident cardiac event but not as a chronic baseline condition.",
    )
    for prefix in ("I12", "I13"):
        current = rules[prefix]
        rules[prefix] = Rule(**{**asdict(current), "secondary_organ": "renal"})
    for prefix in ("I26", "I27", "I28"):
        current = rules[prefix]
        rules[prefix] = Rule(**{**asdict(current), "secondary_organ": "pulmonary"})
    add_rule(
        rules,
        "C380",
        "Malignant neoplasm of heart",
        "cardiovascular",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Anatomically specific cardiac malignancy; broad C38 is not used because it also covers mediastinum and pleura.",
    )
    add_families(
        rules,
        [family("Q", value) for value in range(20, 29)],
        "Congenital malformation of circulatory system",
        "cardiovascular",
        baseline=True,
        incident=False,
        course="congenital",
        rationale="Baseline-only congenital cardiovascular burden; not a new adult incident endpoint.",
    )
    add_rule(
        rules,
        "Z941",
        "Heart transplant status",
        "cardiovascular",
        baseline=True,
        incident=False,
        course="status",
        rationale="Transplant status establishes severe prevalent cardiovascular disease but is not an incident disease code.",
    )
    add_families(
        rules,
        ["I95", "I97", "I99"],
        "Other, postprocedural or unspecified circulatory condition",
        "cardiovascular",
        baseline=False,
        incident=False,
        course="acute_nonspecific_or_secondary",
        rationale="Hypotension, postprocedural and unspecified circulatory codes are retained only in broad morbidity sensitivity.",
    )
    add_rule(
        rules,
        "I98",
        "Other circulatory disorder in disease classified elsewhere",
        "",
        baseline=False,
        incident=False,
        course="secondary_or_ambiguous",
        rationale="The parent family spans heterogeneous secondary manifestations; only anatomically specific hepatic subtypes I982/I983 are assigned below.",
        review_status="reviewed_excluded_ambiguous_parent",
    )
    add_rule(
        rules,
        "I84",
        "Haemorrhoids",
        "",
        baseline=False,
        incident=False,
        course="out_of_target_scope",
        rationale="Haemorrhoidal disease is not treated as target-organ cardiovascular disease.",
        review_status="reviewed_excluded_out_of_target_scope",
    )
    add_rule(
        rules,
        "I81",
        "Portal vein thrombosis",
        "hepatic",
        secondary="cardiovascular",
        baseline=True,
        incident=True,
        course="chronic_or_established_event",
        rationale="Portal-vein thrombosis is assigned to hepatic disease with vascular involvement retained as secondary.",
    )
    add_rule(
        rules,
        "I83",
        "Varicose veins of lower extremities",
        "cardiovascular",
        baseline=False,
        incident=False,
        course="chronic_low_specificity",
        rationale="Varicose-vein coding is retained only in broad vascular sensitivity and not in the main cardiovascular endpoint.",
    )
    add_rule(
        rules,
        "I85",
        "Oesophageal varices",
        "hepatic",
        secondary="cardiovascular",
        baseline=True,
        incident=True,
        course="chronic_or_established_event",
        rationale="Oesophageal varices are a specific manifestation of portal hypertensive liver disease.",
    )
    add_families(
        rules,
        ["I86", "I88", "I89"],
        "Ambiguous variceal, lymph-node or lymphatic disorder",
        "",
        baseline=False,
        incident=False,
        course="ambiguous_or_out_of_target_scope",
        rationale="These families do not identify a unique target organ and are excluded from the seven-organ main analysis.",
        review_status="reviewed_excluded_ambiguous_or_out_of_scope",
    )

    # Pulmonary. Acute respiratory infections and procedural complications are
    # explicitly excluded from the main chronic/incident endpoint.
    add_families(
        rules,
        [family("C", value) for value in range(30, 35)],
        "Malignant neoplasm of respiratory or intrathoracic organ",
        "pulmonary",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Respiratory-tract and lung malignancy within WHO ICD-10 C30-C34.",
    )
    add_rule(
        rules,
        "C38",
        "Ambiguous malignant neoplasm of heart, mediastinum or pleura",
        "",
        baseline=False,
        incident=False,
        course="ambiguous",
        malignancy=True,
        rationale="Unsubtyped C38 is not assigned because it spans cardiovascular and pulmonary anatomy.",
        review_status="reviewed_excluded_ambiguous_anatomy",
    )
    add_families(
        rules,
        [f"C38{value}" for value in (1, 2, 3, 4, 8, 9)],
        "Malignant neoplasm of mediastinum or pleura",
        "pulmonary",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Subtyped C38 code identifies mediastinal or pleural anatomy.",
    )
    add_families(
        rules,
        [family("J", value) for value in list(range(9, 19)) + list(range(20, 23))],
        "Acute lower respiratory infection",
        "pulmonary",
        baseline=False,
        incident=False,
        course="acute",
        rationale="Influenza, pneumonia, acute bronchitis and bronchiolitis are retained only in broad morbidity sensitivity.",
    )
    add_families(
        rules,
        [family("J", value) for value in range(40, 48)],
        "Chronic lower respiratory disease",
        "pulmonary",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="WHO ICD-10 chronic lower respiratory disease block.",
    )
    add_families(
        rules,
        [family("J", value) for value in range(60, 71)],
        "Lung disease due to external agents",
        "pulmonary",
        baseline=True,
        incident=True,
        course="mixed",
        rationale="Established pneumoconiosis, hypersensitivity and exposure-related lung disease.",
    )
    add_rule(
        rules,
        "J69",
        "Pneumonitis due to solids and liquids",
        "pulmonary",
        baseline=False,
        incident=False,
        course="acute",
        rationale="Aspiration pneumonitis is excluded from the main chronic endpoint.",
    )
    for prefix, title, eligible, course in (
        ("J80", "Adult respiratory distress syndrome", False, "acute"),
        ("J81", "Pulmonary oedema", False, "acute_or_secondary"),
        ("J82", "Pulmonary eosinophilia", True, "mixed"),
        ("J84", "Other interstitial pulmonary disease", True, "chronic"),
        ("J85", "Abscess of lung and mediastinum", False, "acute"),
        ("J86", "Pyothorax", False, "acute"),
        ("J90", "Pleural effusion", False, "secondary_manifestation"),
        ("J91", "Pleural effusion in conditions classified elsewhere", False, "secondary_manifestation"),
        ("J92", "Pleural plaque", True, "chronic"),
        ("J93", "Pneumothorax", False, "acute"),
        ("J94", "Other pleural condition", True, "mixed"),
        ("J95", "Postprocedural respiratory disorder", False, "iatrogenic"),
        ("J96", "Respiratory failure", True, "mixed"),
        ("J98", "Other respiratory disorder", True, "mixed"),
        ("J99", "Respiratory disorder in disease classified elsewhere", True, "secondary_manifestation"),
    ):
        add_rule(
            rules,
            prefix,
            title,
            "pulmonary",
            baseline=eligible,
            incident=eligible,
            course=course,
            rationale=(
                "Clinically persistent pulmonary disease eligible for the main endpoint."
                if eligible
                else "Retained only in broad morbidity sensitivity because it is acute, procedural or a nonspecific manifestation."
            ),
        )
    add_families(
        rules,
        [family("Q", value) for value in range(30, 35)],
        "Congenital malformation of respiratory system",
        "pulmonary",
        baseline=True,
        incident=False,
        course="congenital",
        rationale="Baseline-only congenital pulmonary burden.",
    )
    add_rule(
        rules,
        "Z942",
        "Lung transplant status",
        "pulmonary",
        baseline=True,
        incident=False,
        course="status",
        rationale="Transplant status establishes severe prevalent pulmonary disease.",
    )
    add_rule(
        rules,
        "Z943",
        "Heart and lung transplant status",
        "pulmonary",
        secondary="cardiovascular",
        baseline=True,
        incident=False,
        course="status",
        rationale="Combined transplant status contributes to pulmonary and cardiovascular baseline burden.",
    )

    # Hepatic.
    add_rule(
        rules,
        "B18",
        "Chronic viral hepatitis",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Important chronic liver disease omitted from the submitted organ map.",
    )
    add_rule(
        rules,
        "C22",
        "Malignant neoplasm of liver and intrahepatic bile ducts",
        "hepatic",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Anatomically specific hepatic malignancy.",
    )
    add_rule(
        rules,
        "K70",
        "Alcoholic liver disease",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic_or_mixed",
        rationale="Established liver disease with persistent prognostic burden.",
    )
    add_rule(
        rules,
        "K71",
        "Toxic liver disease, unspecified subtype",
        "hepatic",
        baseline=False,
        incident=False,
        course="mixed",
        rationale="Unsubtyped toxic liver disease is retained only in broad sensitivity; chronic subtypes are eligible below.",
    )
    add_families(
        rules,
        ["K713", "K714", "K715", "K717"],
        "Chronic toxic liver disease subtype",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Subtype explicitly identifies chronic hepatitis, fibrosis or cirrhosis.",
    )
    add_rule(
        rules,
        "K72",
        "Hepatic failure, unspecified subtype",
        "hepatic",
        baseline=False,
        incident=False,
        course="mixed",
        rationale="Unsubtyped hepatic failure is retained in broad sensitivity; chronic subtype K721 is eligible.",
    )
    add_rule(
        rules,
        "K721",
        "Chronic hepatic failure",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Explicit chronic hepatic failure.",
    )
    add_families(
        rules,
        ["K73", "K74"],
        "Chronic hepatitis, fibrosis or cirrhosis",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Core chronic liver disease endpoint.",
    )
    add_rule(
        rules,
        "K75",
        "Other inflammatory liver disease, unspecified subtype",
        "hepatic",
        baseline=False,
        incident=False,
        course="mixed",
        rationale="Abscess and acute inflammatory subtypes are not main chronic endpoints; explicit chronic subtypes are eligible below.",
    )
    add_families(
        rules,
        ["K753", "K754", "K758", "K759"],
        "Chronic or persistent inflammatory liver disease subtype",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic_or_mixed",
        rationale="Includes granulomatous, autoimmune and other persistent inflammatory liver disease.",
    )
    add_rule(
        rules,
        "K76",
        "Other disease of liver",
        "hepatic",
        baseline=True,
        incident=True,
        course="chronic_or_mixed",
        rationale="Includes fatty liver, portal hypertension and other established liver disorders.",
    )
    add_families(
        rules,
        ["K762", "K763"],
        "Acute necrosis or infarction of liver",
        "hepatic",
        baseline=False,
        incident=False,
        course="acute",
        rationale="Acute hepatic injury retained only in broad morbidity sensitivity.",
    )
    add_rule(
        rules,
        "K77",
        "Liver disorder in disease classified elsewhere",
        "hepatic",
        baseline=True,
        incident=True,
        course="secondary_manifestation",
        rationale="Explicit hepatic involvement, although the underlying disease is classified elsewhere.",
    )
    add_rule(
        rules,
        "Q44",
        "Congenital malformation of gallbladder, bile ducts or liver",
        "hepatic",
        baseline=True,
        incident=False,
        course="congenital",
        rationale="Baseline-only congenital hepatobiliary burden.",
    )
    add_rule(
        rules,
        "Z944",
        "Liver transplant status",
        "hepatic",
        baseline=True,
        incident=False,
        course="status",
        rationale="Transplant status establishes severe prevalent hepatic disease.",
    )
    add_families(
        rules,
        ["I982", "I983"],
        "Oesophageal varices in disease classified elsewhere",
        "hepatic",
        secondary="cardiovascular",
        baseline=True,
        incident=True,
        course="chronic_or_established_event",
        rationale="Oesophageal varices are a clinically specific manifestation of portal hypertensive liver disease; vascular involvement is retained as secondary.",
    )

    # Immune. The submitted map omitted common systemic autoimmune disease.
    add_families(
        rules,
        [family("B", value) for value in range(20, 25)],
        "Human immunodeficiency virus disease",
        "immune",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Chronic acquired immune deficiency with persistent immune-system burden.",
    )
    add_rule(
        rules,
        "C37",
        "Malignant neoplasm of thymus",
        "immune",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Thymic malignancy assigned to immune system.",
    )
    add_families(
        rules,
        [family("C", value) for value in range(81, 97)],
        "Lymphoid, haematopoietic or related malignancy",
        "immune",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="WHO ICD-10 C81-C96 lymphoid and haematopoietic malignancy block.",
    )
    add_families(
        rules,
        [family("D", value) for value in range(80, 90)],
        "Disorder involving the immune mechanism",
        "immune",
        baseline=True,
        incident=True,
        course="chronic_or_mixed",
        rationale="WHO ICD-10 D80-D89 immune-mechanism block.",
    )
    current = rules["D86"]
    rules["D86"] = Rule(**{**asdict(current), "secondary_organ": "pulmonary"})
    add_families(
        rules,
        ["M05", "M06", "M07", "M08", "M09"],
        "Immune-mediated inflammatory polyarthropathy",
        "immune",
        secondary="musculoskeletal",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Systemic immune-mediated arthritis; musculoskeletal involvement retained as secondary sensitivity.",
    )
    add_families(
        rules,
        [family("M", value) for value in range(30, 37)],
        "Systemic connective-tissue autoimmune disorder",
        "immune",
        secondary="musculoskeletal",
        baseline=True,
        incident=True,
        course="chronic",
        rationale="Vasculitis, lupus, systemic sclerosis, myositis and related systemic autoimmune disease.",
    )
    add_rule(
        rules,
        "Z9484",
        "Stem-cell transplant status",
        "immune",
        baseline=True,
        incident=False,
        course="status",
        rationale="Baseline-only marker of severe immune/haematopoietic disease burden.",
    )
    add_families(
        rules,
        ["Z947", "Z948"],
        "Ambiguous or out-of-scope transplant status",
        "",
        baseline=False,
        incident=False,
        course="status",
        rationale="Corneal or unspecified transplant status cannot be assigned to one of the seven target organs.",
        review_status="reviewed_excluded_ambiguous_or_out_of_scope",
    )

    # Metabolic and endocrine.
    add_families(
        rules,
        [family("C", value) for value in range(73, 76)],
        "Malignant neoplasm of endocrine gland",
        "metabolic",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Endocrine-gland malignancy assigned to metabolic/endocrine system.",
    )
    metabolic_blocks = (
        (0, 7, "Thyroid disorder"),
        (10, 14, "Diabetes mellitus"),
        (16, 16, "Other disorder of pancreatic internal secretion"),
        (20, 35, "Other endocrine-gland disorder"),
        (40, 46, "Malnutrition"),
        (50, 64, "Nutritional deficiency"),
        (65, 68, "Obesity and hyperalimentation"),
        (70, 90, "Metabolic disorder"),
    )
    for start, end, title in metabolic_blocks:
        add_families(
            rules,
            [family("E", value) for value in range(start, end + 1)],
            title,
            "metabolic",
            baseline=True,
            incident=True,
            course="chronic_or_mixed",
            rationale="WHO ICD-10 endocrine, nutritional or metabolic disease with persistent clinical burden.",
        )
    add_rule(
        rules,
        "E15",
        "Nondiabetic hypoglycaemic coma",
        "metabolic",
        baseline=False,
        incident=False,
        course="acute",
        rationale="Acute metabolic event retained only in broad morbidity sensitivity.",
    )
    add_families(
        rules,
        ["E16", "E86", "E87"],
        "Acute or nonspecific metabolic disturbance",
        "metabolic",
        baseline=False,
        incident=False,
        course="acute_or_nonspecific",
        rationale="Hypoglycaemia, volume depletion and electrolyte/acid-base disturbances are retained only in broad morbidity sensitivity.",
    )

    # Musculoskeletal. Immune-mediated M05-M09 and M30-M36 are primary immune
    # rules above. M80/M81/M83 are primary musculoskeletal with metabolic
    # secondary assignment, eliminating the previous main-analysis double count.
    add_families(
        rules,
        ["C40", "C41", "C49"],
        "Malignant neoplasm of bone, cartilage, connective or soft tissue",
        "musculoskeletal",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Anatomically specific bone, cartilage, connective-tissue or soft-tissue malignancy.",
    )
    add_families(
        rules,
        [family("M", value) for value in range(0, 4)],
        "Infectious arthropathy",
        "musculoskeletal",
        baseline=False,
        incident=False,
        course="acute",
        rationale="Acute infectious arthropathy retained only in broad morbidity sensitivity.",
    )
    for start, end, title in (
        (10, 25, "Inflammatory, degenerative or other joint disease"),
        (40, 54, "Dorsopathy"),
        (60, 68, "Muscle, synovium or tendon disorder"),
        (70, 79, "Other soft-tissue disorder"),
        (80, 99, "Bone, cartilage or other musculoskeletal disorder"),
    ):
        add_families(
            rules,
            [family("M", value) for value in range(start, end + 1)],
            title,
            "musculoskeletal",
            baseline=True,
            incident=True,
            course="chronic_or_mixed",
            rationale="WHO ICD-10 musculoskeletal/connective-tissue disease with persistent or recurrent burden.",
        )
    add_rule(
        rules,
        "M54",
        "Dorsalgia without specified structural disease",
        "musculoskeletal",
        baseline=False,
        incident=False,
        course="symptom_like",
        rationale="Back-pain coding without structural diagnosis is excluded from the main disease endpoint.",
    )
    add_rule(
        rules,
        "M96",
        "Postprocedural musculoskeletal disorder",
        "musculoskeletal",
        baseline=False,
        incident=False,
        course="iatrogenic",
        rationale="Postprocedural complications are retained only in broad morbidity sensitivity.",
    )
    for prefix in ("M80", "M81", "M83"):
        current = rules[prefix]
        rules[prefix] = Rule(**{**asdict(current), "secondary_organ": "metabolic"})
    add_families(
        rules,
        [family("Q", value) for value in range(65, 80)],
        "Congenital musculoskeletal malformation or deformation",
        "musculoskeletal",
        baseline=True,
        incident=False,
        course="congenital",
        rationale="Baseline-only congenital musculoskeletal burden.",
    )

    # Renal/urinary. Acute kidney injury, acute pyelonephritis, stones and renal
    # colic are broad-sensitivity diagnoses, not main chronic incident endpoints.
    add_families(
        rules,
        [family("C", value) for value in range(64, 69)],
        "Malignant neoplasm of urinary tract",
        "renal",
        baseline=True,
        incident=True,
        course="malignancy",
        malignancy=True,
        rationale="Retains the submitted renal/urinary scope; manuscript must define that bladder and urinary-tract cancers are included.",
    )
    renal_eligible = {
        "N01": "Rapidly progressive nephritic syndrome",
        "N03": "Chronic nephritic syndrome",
        "N04": "Nephrotic syndrome",
        "N05": "Unspecified nephritic syndrome",
        "N07": "Hereditary nephropathy",
        "N08": "Glomerular disorder in disease classified elsewhere",
        "N11": "Chronic tubulo-interstitial nephritis",
        "N12": "Tubulo-interstitial nephritis, unspecified",
        "N13": "Obstructive and reflux uropathy",
        "N14": "Drug- or heavy-metal-induced tubulo-interstitial disorder",
        "N16": "Tubulo-interstitial disorder in disease classified elsewhere",
        "N18": "Chronic kidney disease",
        "N19": "Unspecified kidney failure",
        "N25": "Disorder resulting from impaired renal tubular function",
        "N26": "Unspecified contracted kidney",
        "N27": "Small kidney of unknown cause",
        "N28": "Other disorder of kidney and ureter",
        "N29": "Other kidney/ureter disorder in disease classified elsewhere",
    }
    for prefix, title in renal_eligible.items():
        add_rule(
            rules,
            prefix,
            title,
            "renal",
            baseline=True,
            incident=True,
            course="chronic_or_mixed",
            rationale="Established renal or upper-urinary disease suitable for the main endpoint.",
        )
    renal_excluded = {
        "N00": "Acute nephritic syndrome",
        "N02": "Recurrent or persistent haematuria",
        "N06": "Isolated proteinuria",
        "N10": "Acute pyelonephritis",
        "N15": "Other renal tubulo-interstitial infection",
        "N17": "Acute kidney injury",
        "N20": "Kidney or ureteric calculus",
        "N21": "Lower urinary-tract calculus",
        "N22": "Calculus in disease classified elsewhere",
        "N23": "Unspecified renal colic",
    }
    for prefix, title in renal_excluded.items():
        add_rule(
            rules,
            prefix,
            title,
            "renal",
            baseline=False,
            incident=False,
            course="acute_or_nonspecific",
            rationale="Retained only in broad morbidity sensitivity; excluded from the main chronic renal endpoint.",
        )
    add_families(
        rules,
        [family("Q", value) for value in range(60, 65)],
        "Congenital malformation of urinary system",
        "renal",
        baseline=True,
        incident=False,
        course="congenital",
        rationale="Baseline-only congenital renal/urinary burden.",
    )
    add_rule(
        rules,
        "Z940",
        "Kidney transplant status",
        "renal",
        baseline=True,
        incident=False,
        course="status",
        rationale="Transplant status establishes severe prevalent renal disease.",
    )
    add_rule(
        rules,
        "Z992",
        "Dependence on renal dialysis",
        "renal",
        baseline=True,
        incident=False,
        course="status",
        rationale="Dialysis dependence establishes severe prevalent renal disease.",
    )

    return rules


def sorted_rules(rules: dict[str, Rule]) -> list[Rule]:
    return sorted(rules.values(), key=lambda item: (-len(item.icd_prefix), item.icd_prefix))


def classify(code: str, ordered: list[Rule]) -> Rule | None:
    for rule in ordered:
        if code.startswith(rule.icd_prefix):
            return rule
    return None


def cross_primary_prefix_overlaps(rules: dict[str, Rule]) -> list[tuple[str, str, str, str]]:
    primary = [rule for rule in rules.values() if rule.primary_organ]
    overlaps: list[tuple[str, str, str, str]] = []
    for index, left in enumerate(primary):
        for right in primary[index + 1 :]:
            if left.primary_organ == right.primary_organ:
                continue
            if left.icd_prefix.startswith(right.icd_prefix) or right.icd_prefix.startswith(left.icd_prefix):
                overlaps.append(
                    (
                        left.icd_prefix,
                        left.primary_organ,
                        right.icd_prefix,
                        right.primary_organ,
                    )
                )
    return sorted(overlaps)


def load_v1(path: Path) -> dict[str, tuple[str, ...]]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"organ_system", "icd_prefix_normalized"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in v1 mapping: {required - set(frame.columns)}")
    result: dict[str, tuple[str, ...]] = {}
    for organ in ORGANS:
        values = frame.loc[frame["organ_system"].eq(organ), "icd_prefix_normalized"]
        result[organ] = tuple(sorted(set(values), key=lambda value: (-len(value), value)))
    return result


def classify_v1(code: str, mapping: dict[str, tuple[str, ...]]) -> str:
    return ";".join(organ for organ in ORGANS if code.startswith(mapping[organ]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_raw(
    raw_path: Path,
    cohort_ids: set[str],
    ordered: list[Rule],
    v1_mapping: dict[str, tuple[str, ...]],
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    row_counts: Counter[str] = Counter()
    descriptions: dict[str, str] = {}
    v1_cache: dict[str, str] = {}
    v2_cache: dict[str, Rule | None] = {}
    participants_changed: dict[str, set[str]] = defaultdict(set)
    per_organ = {
        organ: Counter(
            v1_rows=0,
            v2_primary_rows=0,
            v2_secondary_rows=0,
            baseline_eligible_rows=0,
            incident_eligible_rows=0,
            rows_added_to_primary=0,
            rows_removed_from_v1=0,
        )
        for organ in ORGANS
    }
    source_rows = 0
    cohort_rows = 0
    valid_rows = 0

    reader = pd.read_csv(
        raw_path,
        usecols=["Participant ID", "ICD", "Disease_Description_Feature"],
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        source_rows += len(chunk)
        chunk["Participant ID"] = (
            chunk["Participant ID"]
            .fillna("")
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        chunk = chunk.loc[chunk["Participant ID"].isin(cohort_ids)].copy()
        cohort_rows += len(chunk)
        chunk["code"] = (
            chunk["ICD"]
            .fillna("")
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
            .str.strip()
        )
        chunk = chunk.loc[chunk["code"].ne("")].copy()
        valid_rows += len(chunk)
        counts = chunk["code"].value_counts()
        row_counts.update(counts.to_dict())
        for row in chunk.loc[
            chunk["Disease_Description_Feature"].notna(),
            ["code", "Disease_Description_Feature"],
        ].drop_duplicates("code").itertuples(index=False):
            descriptions.setdefault(row.code, str(row.Disease_Description_Feature).strip())

        unique_codes = chunk["code"].unique().tolist()
        for code in unique_codes:
            if code not in v1_cache:
                v1_cache[code] = classify_v1(code, v1_mapping)
                v2_cache[code] = classify(code, ordered)
        chunk["v1_organs"] = chunk["code"].map(v1_cache)
        chunk["v2_rule"] = chunk["code"].map(v2_cache)
        chunk["v2_primary"] = chunk["v2_rule"].map(
            lambda rule: rule.primary_organ if rule is not None else ""
        )
        chunk["v2_secondary"] = chunk["v2_rule"].map(
            lambda rule: rule.secondary_organ if rule is not None else ""
        )
        chunk["v2_baseline"] = chunk["v2_rule"].map(
            lambda rule: bool(rule and rule.baseline_chronic_eligible)
        )
        chunk["v2_incident"] = chunk["v2_rule"].map(
            lambda rule: bool(rule and rule.incident_endpoint_eligible)
        )

        for organ in ORGANS:
            old = chunk["v1_organs"].str.split(";").map(lambda values: organ in values)
            primary = chunk["v2_primary"].eq(organ)
            secondary = chunk["v2_secondary"].str.split(";").map(lambda values: organ in values)
            per_organ[organ]["v1_rows"] += int(old.sum())
            per_organ[organ]["v2_primary_rows"] += int(primary.sum())
            per_organ[organ]["v2_secondary_rows"] += int(secondary.sum())
            per_organ[organ]["baseline_eligible_rows"] += int((primary & chunk["v2_baseline"]).sum())
            per_organ[organ]["incident_eligible_rows"] += int((primary & chunk["v2_incident"]).sum())
            added = primary & ~old
            removed = old & ~primary
            per_organ[organ]["rows_added_to_primary"] += int(added.sum())
            per_organ[organ]["rows_removed_from_v1"] += int(removed.sum())
            changed = added | removed
            if changed.any():
                participants_changed[organ].update(
                    chunk.loc[changed, "Participant ID"].dropna().astype(str).tolist()
                )
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"chunks={chunk_number}; source_rows={source_rows:,}; valid_icd_rows={valid_rows:,}",
                flush=True,
            )

    observed_rows: list[dict[str, object]] = []
    for code in sorted(row_counts):
        rule = v2_cache[code]
        old = v1_cache[code]
        new = rule.primary_organ if rule else ""
        observed_rows.append(
            {
                "icd_code_normalized": code,
                "description": descriptions.get(code, ""),
                "source_row_count": int(row_counts[code]),
                "v1_organ_systems": old,
                "v2_primary_organ": new,
                "v2_secondary_organ": rule.secondary_organ if rule else "",
                "baseline_chronic_eligible": bool(rule and rule.baseline_chronic_eligible),
                "incident_endpoint_eligible": bool(rule and rule.incident_endpoint_eligible),
                "recurrent_endpoint_eligible": bool(rule and rule.recurrent_endpoint_eligible),
                "acute_or_chronic": rule.acute_or_chronic if rule else "outside_seven_organ_scope",
                "malignancy_flag": bool(rule and rule.malignancy_flag),
                "matched_rule_prefix": rule.icd_prefix if rule else "",
                "review_status": rule.review_status if rule else "outside_prespecified_seven_organ_scope",
                "classification_changed": old != new,
            }
        )
    observed = pd.DataFrame(observed_rows).sort_values(
        ["source_row_count", "icd_code_normalized"], ascending=[False, True]
    )
    impact_rows = []
    for organ in ORGANS:
        impact_rows.append(
            {
                "organ_system": organ,
                **{key: int(value) for key, value in per_organ[organ].items()},
                "participants_with_any_primary_classification_change": len(participants_changed[organ]),
            }
        )
    impact = pd.DataFrame(impact_rows)
    scan_counts = {
        "source_rows_scanned": int(source_rows),
        "age_cohort_source_rows": int(cohort_rows),
        "valid_icd_rows": int(valid_rows),
        "observed_unique_codes": int(len(observed)),
    }
    return observed, impact, scan_counts


def build(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = build_rules()
    ordered = sorted_rules(rules)
    v1_mapping = load_v1(args.v1_mapping)
    cohort = pd.read_parquet(args.cohort, columns=["participant_id"])
    cohort_ids = set(
        cohort["participant_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if len(rules) != len(set(rules)):
        raise RuntimeError("Duplicate v2 prefixes")
    if any(not rule.icd_prefix for rule in ordered):
        raise RuntimeError("Empty v2 prefix")
    if any(rule.primary_organ and rule.primary_organ not in ORGANS for rule in ordered):
        raise RuntimeError("Unknown v2 organ")
    cross_primary_overlaps = cross_primary_prefix_overlaps(rules)
    if cross_primary_overlaps:
        raise RuntimeError(f"Cross-primary hierarchical prefix overlaps: {cross_primary_overlaps}")

    observed, impact, scan_counts = scan_raw(
        args.raw_icd,
        cohort_ids,
        ordered,
        v1_mapping,
        args.chunksize,
    )
    stored = pd.read_csv(args.observed_codes, dtype="string", keep_default_na=False)
    stored_codes = set(stored["icd_code_normalized"])
    scanned_codes = set(observed["icd_code_normalized"])
    if stored_codes != scanned_codes:
        raise RuntimeError(
            "Observed-code universe differs from the frozen Revision-0710 export: "
            f"missing_from_scan={len(stored_codes - scanned_codes)}, "
            f"new_in_scan={len(scanned_codes - stored_codes)}"
        )

    rule_frame = pd.DataFrame([asdict(rule) for rule in sorted(rules.values(), key=lambda x: x.icd_prefix)])
    mapping = rule_frame.loc[
        rule_frame["primary_organ"].ne(""),
        ["primary_organ", "icd_prefix"],
    ].rename(columns={"primary_organ": "organ_system", "icd_prefix": "icd_prefix_normalized"})
    mapping = mapping.sort_values(["organ_system", "icd_prefix_normalized"]).reset_index(drop=True)

    codebook_path = args.output_dir / "codebook_v2.csv"
    mapping_path = args.output_dir / "mapping_primary.csv"
    observed_path = args.output_dir / "observed_code_audit.csv"
    impact_path = args.output_dir / "impact_summary.csv"
    rule_frame.to_csv(codebook_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    observed.to_csv(observed_path, index=False)
    impact.to_csv(impact_path, index=False)

    high_frequency_unreviewed = observed.loc[
        observed["source_row_count"].ge(100)
        & observed["review_status"].eq("outside_prespecified_seven_organ_scope")
        & observed["icd_code_normalized"].str.match(
            r"^(B1[8-9]|B2[0-4]|C(?:22|3[0-8]|4[019]|6[4-8]|7[3-5]|8[1-9]|9[0-6])|D8[0-9]|E|I|J(?:0[9]|1[0-8]|2[0-2]|4[0-7]|6[0-9]|70|8[0-6]|9[0-689])|K7[0-7]|M|N(?:0[0-8]|1[0-9]|2[0-9])|Q(?:2[0-8]|3[0-4]|44|6[0-9]|7[0-9])|Z94|Z992)"
        )
    ]
    if not high_frequency_unreviewed.empty:
        examples = high_frequency_unreviewed[
            ["icd_code_normalized", "description", "source_row_count"]
        ].head(20).to_dict("records")
        raise RuntimeError(f"High-frequency target-scope codes remain unreviewed: {examples}")

    # Frozen semantic checks for the specific errors identified in v1.
    checks = {
        "C49_primary_musculoskeletal": classify("C499", ordered).primary_organ == "musculoskeletal",
        "M80_single_primary": classify("M800", ordered).primary_organ == "musculoskeletal",
        "M80_metabolic_secondary": classify("M800", ordered).secondary_organ == "metabolic",
        "M05_primary_immune": classify("M059", ordered).primary_organ == "immune",
        "M30_primary_immune": classify("M300", ordered).primary_organ == "immune",
        "J18_not_main_endpoint": not classify("J189", ordered).incident_endpoint_eligible,
        "N17_not_main_endpoint": not classify("N179", ordered).incident_endpoint_eligible,
        "B18_hepatic": classify("B189", ordered).primary_organ == "hepatic",
        "E78_metabolic": classify("E789", ordered).primary_organ == "metabolic",
        "C64_renal": classify("C64", ordered).primary_organ == "renal",
        "N299_valid": classify("N299", ordered).primary_organ == "renal",
        "no_N299C64_prefix": "N299C64" not in rules,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Semantic checks failed: {checks}")

    output_paths = [codebook_path, mapping_path, observed_path, impact_path]
    manifest = {
        "schema_version": "revision0710.icd_codebook_v2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "Clinically structured ICD-10 organ mapping and eligibility audit; no downstream cache or model was modified.",
        "inputs": {
            "v1_mapping": {"path": str(args.v1_mapping.resolve()), "sha256": sha256_file(args.v1_mapping)},
            "observed_codes": {"path": str(args.observed_codes.resolve()), "sha256": sha256_file(args.observed_codes)},
            "age_model_cohort": {"path": str(args.cohort.resolve()), "sha256": sha256_file(args.cohort)},
            "raw_icd": {"path": str(args.raw_icd.resolve()), "sha256": sha256_file(args.raw_icd)},
        },
        "counts": {
            **scan_counts,
            "rule_rows": int(len(rule_frame)),
            "primary_mapping_rows": int(len(mapping)),
            "observed_codes_with_primary_organ": int(observed["v2_primary_organ"].ne("").sum()),
            "observed_codes_baseline_eligible": int(observed["baseline_chronic_eligible"].sum()),
            "observed_codes_incident_eligible": int(observed["incident_endpoint_eligible"].sum()),
            "observed_codes_primary_changed_vs_v1": int(observed["classification_changed"].sum()),
            "high_frequency_target_scope_unreviewed": int(len(high_frequency_unreviewed)),
            "multi_primary_prefixes": int(len(cross_primary_overlaps)),
            "malformed_prefixes": 0,
        },
        "semantic_checks": checks,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Revision-0710 ICD codebook v2 audit.")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--v1-mapping", type=Path, default=DEFAULT_V1)
    parser.add_argument("--observed-codes", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--raw-icd", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.v1_mapping, args.observed_codes, args.cohort, args.raw_icd):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = build(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
