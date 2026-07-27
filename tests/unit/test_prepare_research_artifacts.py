from bts_nvs.data.prepare_research_artifacts import (
    RESEARCH_HOLDOUT_NAME,
)


def test_research_artifact_name_is_distinct_from_historical_holdout() -> None:
    assert RESEARCH_HOLDOUT_NAME == "holdout_research_v3.json"
    assert RESEARCH_HOLDOUT_NAME != "holdout.json"
