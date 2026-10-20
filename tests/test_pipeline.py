"""
Basic structural tests for the RECS Energy Prediction Pipeline.

These tests verify the pipeline module's structure and configuration
without running the full training (which requires ~110 MB of CSV data
and XGBoost training).
"""

import os
import sys

# Add project root to path so we can import pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline  # noqa: E402


def test_column_maps_exist_for_all_years():
    """Every survey year should have a column mapping defined."""
    for year in [2005, 2009, 2015, 2020]:
        assert year in pipeline.COL_MAP, f"Missing COL_MAP for year {year}"
        assert 'kwh' in pipeline.COL_MAP[year], f"Missing kwh mapping for {year}"


def test_target_is_kwh():
    """The prediction target should be annual electricity consumption (kWh)."""
    assert pipeline.TARGET == 'kwh'


def test_survey_year_not_in_features():
    """
    survey_year must NOT be a model feature.

    Since the test set is the held-out 2020 cycle, using survey_year as a
    feature would let the model memorize year-specific patterns instead of
    learning generalizable relationships.
    """
    assert 'survey_year' not in pipeline.FEATURES, (
        "survey_year should not be a model feature (temporal leakage risk)"
    )


def test_categorical_cols_defined():
    """Categorical columns should be explicitly listed for encoding."""
    expected = [
        'heat_equip', 'heat_fuel', 'housing_type', 'region',
        'climate_region', 'ac_type', 'division',
    ]
    for col in expected:
        assert col in pipeline.CATEGORICAL_COLS, f"Missing categorical col: {col}"


def test_feature_count():
    """The model should use a reasonable number of features (> 10)."""
    assert len(pipeline.FEATURES) > 10, "Too few features defined"


def test_files_mapping_exists():
    """Each survey year should map to a source CSV filename."""
    for year in [2005, 2009, 2015, 2020]:
        assert year in pipeline.FILES, f"Missing FILES entry for year {year}"
        assert pipeline.FILES[year].endswith('.csv')


def test_missing_codes_defined():
    """RECS missing-value codes should be defined for NaN conversion."""
    assert -2 in pipeline.RECS_MISSING
    assert -8 in pipeline.RECS_MISSING
    assert -9 in pipeline.RECS_MISSING
    assert -99 in pipeline.RECS_MISSING