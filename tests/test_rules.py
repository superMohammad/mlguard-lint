"""Regression tests for MLGuard rules.

Each fixture under tests/notebooks/ must surface at least its target rule, the
clean controls must stay silent, and the targeted heuristics (MLG003, MLG015)
must not fire on their negative controls.
"""
import pathlib

import pytest

from mlguard_lint.scanner import scan_code, scan_path

NB_DIR = pathlib.Path(__file__).parent / "notebooks"

# Fixture -> rule ids that must be present in its diagnostics.
EXPECTED_PRESENT = {
    "16_imputer_assigned_instance.ipynb": {"MLG001"},
    "17_no_test_set.ipynb": {"MLG012"},
    "18_resample_test_set.ipynb": {"MLG013"},
    "19_timeseries_random_cv.ipynb": {"MLG014"},
    "20_target_in_features.ipynb": {"MLG015"},
    "21_target_encoder_fit_transform.ipynb": {"MLG011"},
    "22_ordinal_encoder_feature.ipynb": {"MLG009"},
    "24_scale_before_cv.ipynb": {"MLG002"},
    "25_resample_before_split.ipynb": {"MLG004"},
    "26_split_without_stratify.ipynb": {"MLG005"},
    "27_missing_random_state.ipynb": {"MLG006"},
    "28_group_leakage_split.ipynb": {"MLG007"},
    "29_gridsearch_best_score.ipynb": {"MLG008"},
    "30_test_set_reused.ipynb": {"MLG010"},
    "31_refit_on_test.ipynb": {"MLG016"},
    "32_auc_on_hard_preds.ipynb": {"MLG017"},
    "33_global_stat_feature.ipynb": {"MLG018"},
    "34_micro_average.ipynb": {"MLG019"},
    "35_id_column_feature.ipynb": {"MLG020"},
    "36_upsample_before_split.ipynb": {"MLG021"},
}

CLEAN_FIXTURES = {
    "15_clean_pipeline.ipynb",
    "23_clean_timeseries_pipeline.ipynb",
}


@pytest.mark.parametrize("fixture,expected", sorted(EXPECTED_PRESENT.items()))
def test_expected_rule_present(fixture: str, expected: set) -> None:
    rule_ids = {d.rule_id for d in scan_path(str(NB_DIR / fixture))}
    assert expected <= rule_ids, f"{fixture}: missing {expected - rule_ids} (got {rule_ids})"


@pytest.mark.parametrize("fixture", sorted(CLEAN_FIXTURES))
def test_clean_fixture_silent(fixture: str) -> None:
    diags = scan_path(str(NB_DIR / fixture))
    assert diags == [], f"{fixture}: expected no diagnostics, got {[d.rule_id for d in diags]}"


def test_mlg003_not_fired_on_shared_y_different_x() -> None:
    # Same y name across train/test but a proper held-out X must not be flagged.
    src = "model.fit(X_train, y)\nprint(model.score(X_test, y))\n"
    assert "MLG003" not in {d.rule_id for d in scan_code(src)}


def test_mlg003_fired_on_evaluate_on_training() -> None:
    src = "model.fit(X_train, y_train)\nprint(model.score(X_train, y_train))\n"
    assert "MLG003" in {d.rule_id for d in scan_code(src)}


def test_mlg015_not_fired_when_target_dropped() -> None:
    src = "X = df.drop(columns=['target'])\ny = df['target']\nclf.fit(X, y)\n"
    assert "MLG015" not in {d.rule_id for d in scan_code(src)}


def test_no_parser_or_scanner_errors_across_suite() -> None:
    rule_ids = {d.rule_id for d in scan_path(str(NB_DIR))}
    assert "ERROR" not in rule_ids and "PARSE" not in rule_ids
