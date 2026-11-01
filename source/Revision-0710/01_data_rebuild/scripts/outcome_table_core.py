from __future__ import annotations

import numpy as np
import pandas as pd


MONTH_DAYS = 365.25 / 12.0


def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"missing required column: {column}")
    return pd.to_datetime(frame[column], errors="coerce")


def build_organ_outcomes(
    frame: pd.DataFrame,
    *,
    horizon_months: int = 60,
) -> pd.DataFrame:
    required = {
        "participant_id",
        "index_date",
        "observed_followup_end",
        "death_date",
        "has_prevalent_target",
        "first_incident_target_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"organ outcome frame missing columns: {missing}")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive")

    index_date = _dates(frame, "index_date")
    observed = _dates(frame, "observed_followup_end")
    death = _dates(frame, "death_date")
    target = _dates(frame, "first_incident_target_date")
    prevalent = frame["has_prevalent_target"].fillna(False).astype(bool)
    administrative_end = index_date + pd.to_timedelta(horizon_months * MONTH_DAYS, unit="D")

    observed_valid = observed.notna() & observed.gt(index_date)
    target_valid = target.notna() & target.gt(index_date)
    death_valid = death.notna() & death.gt(index_date)
    observed_explicitly_invalid = observed.notna() & ~observed_valid
    endpoint_evidence = (target_valid | death_valid) & ~observed_explicitly_invalid

    observation_limit = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    observation_limit.loc[observed_valid] = pd.concat(
        [observed.loc[observed_valid], administrative_end.loc[observed_valid]], axis=1
    ).min(axis=1)
    fallback = observed.isna() & endpoint_evidence
    observation_limit.loc[fallback] = administrative_end.loc[fallback]

    death_observed = death_valid & observation_limit.notna() & death.le(observation_limit)
    target_observed = (
        target_valid
        & observation_limit.notna()
        & target.le(observation_limit)
        & (~death_observed | target.le(death))
    )
    at_risk = ~prevalent & observation_limit.notna() & observation_limit.gt(index_date)
    event = at_risk & target_observed
    competing_death = at_risk & ~event & death_observed

    end_date = observation_limit.copy()
    end_date.loc[competing_death] = death.loc[competing_death]
    end_date.loc[event] = target.loc[event]
    duration = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    duration.loc[at_risk] = (
        (end_date.loc[at_risk] - index_date.loc[at_risk]).dt.days / MONTH_DAYS
    ).clip(lower=1e-6)

    reason = pd.Series("no_post_index_followup", index=frame.index, dtype="string")
    reason.loc[prevalent] = "prevalent_target_excluded"
    administrative = at_risk & end_date.eq(administrative_end)
    early_observation = at_risk & ~administrative
    reason.loc[administrative] = "administrative_60m"
    reason.loc[early_observation] = "observed_followup_end"
    reason.loc[competing_death] = "death_before_target"
    reason.loc[event] = "target_event"

    return pd.DataFrame(
        {
            "participant_id": frame["participant_id"].astype("string"),
            "event": event.astype(np.int8),
            "duration_months": duration,
            "competing_death": competing_death.astype(np.int8),
            "at_risk": at_risk.astype(np.int8),
            "censor_reason": reason,
            "outcome_end_date": end_date,
        }
    )


def strict_healthy_reference_mask(
    frame: pd.DataFrame,
    *,
    horizon_months: int = 60,
) -> pd.Series:
    required = {
        "index_date",
        "observed_followup_end",
        "death_date",
        "has_prevalent_any_icd",
        "first_incident_any_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"strict healthy frame missing columns: {missing}")
    index_date = _dates(frame, "index_date")
    observed = _dates(frame, "observed_followup_end")
    death = _dates(frame, "death_date")
    incident = _dates(frame, "first_incident_any_date")
    administrative_end = index_date + pd.to_timedelta(horizon_months * MONTH_DAYS, unit="D")
    prevalent = frame["has_prevalent_any_icd"].fillna(False).astype(bool)
    incident_within = incident.notna() & incident.gt(index_date) & incident.le(administrative_end)
    death_within = death.notna() & death.gt(index_date) & death.le(administrative_end)
    complete = observed.notna() & observed.ge(administrative_end)
    mask = ~prevalent & ~incident_within & ~death_within & complete
    mask.name = "strict_healthy_reference_60m"
    return mask
