import os
import json
import pandas as pd

import numpy as np

from shackett_utils.genomics import mdata_factor_analysis as mfa


def test_mofa_variance_metrics(simple_mdata):
    """Test MOFA variance calculation with different numbers of factors."""
    # Test with different numbers of factors
    factor_values = [2, 4]

    for n_factors in factor_values:
        # Fit MOFA
        mfa._mofa(
            simple_mdata,
            n_factors=n_factors,
            use_layer="log2_centered",
            use_var=None,  # Explicitly disable variable selection
            seed=42,
        )

        # Calculate variance metrics
        metrics = mfa._calculate_mofa_variance_metrics(
            simple_mdata, use_layer="log2_centered"
        )

        # Basic validation of metrics structure
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == {
            "total_variance",
            "modality_variance",
            "raw_tss",
            "raw_ess",
        }
        assert set(metrics["modality_variance"].keys()) == {"rna", "atac"}

        # Validate variance calculations
        for modality in ["rna", "atac"]:
            # Variance explained should be between 0 and 100%
            assert (
                0 <= metrics["modality_variance"][modality] <= 100
            ), f"Variance for {modality} ({metrics['modality_variance'][modality]}%) outside valid range"

            # Raw TSS and ESS should be positive
            assert metrics["raw_tss"][modality] > 0
            assert metrics["raw_ess"][modality] > 0

            # ESS should not exceed TSS
            assert (
                metrics["raw_ess"][modality] <= metrics["raw_tss"][modality] * 1.01
            )  # Allow 1% numerical error

        # Total variance should be weighted average of modality variances
        total_tss = sum(metrics["raw_tss"].values())
        weights = {mod: tss / total_tss for mod, tss in metrics["raw_tss"].items()}
        expected_total = sum(
            metrics["modality_variance"][mod] * weights[mod]
            for mod in metrics["modality_variance"].keys()
        )
        np.testing.assert_allclose(
            metrics["total_variance"], expected_total, rtol=1e-10
        )


def test_mofa_factor_analysis(simple_mdata, tmp_path):
    """Test MOFA factor analysis and variance calculations."""
    # Set up temporary directory for MOFA results
    models_dir = os.path.join(tmp_path, "mofa_models")

    # Run MOFA with enough factors to capture all patterns
    factor_results = mfa.run_mofa_factor_scan(
        simple_mdata,
        factor_range=[
            3,
            5,
        ],  # Test with 3 and 5 factors to ensure we can capture all patterns
        use_layer="log2_centered",
        use_var=None,  # Explicitly disable variable selection
        models_dir=models_dir,
        seed=42,
    )

    # Basic validation of results structure
    assert set(factor_results.keys()) == {3, 5}
    for n_factors in [3, 5]:
        assert "summary_file" in factor_results[n_factors]
        assert os.path.exists(factor_results[n_factors]["summary_file"])

    # Load metrics for each factor value
    metrics = {}
    for n_factors in [3, 5]:
        with open(factor_results[n_factors]["summary_file"], "r") as f:
            metrics[n_factors] = json.load(f)

    # For each factor value, validate the metrics
    for n_factors in [3, 5]:
        # Validate metric structure
        assert isinstance(metrics[n_factors], dict)
        assert set(metrics[n_factors].keys()) == {
            "total_variance",
            "modality_variance",
            "raw_tss",
            "raw_ess",
        }
        assert set(metrics[n_factors]["modality_variance"].keys()) == {"rna", "atac"}

        # Validate variance calculations with realistic bounds
        for modality in ["rna", "atac"]:
            # With 3+ factors, we should explain decent variance in both modalities
            assert (
                20 <= metrics[n_factors]["modality_variance"][modality] <= 100
            ), f"Variance for {modality} ({metrics[n_factors]['modality_variance'][modality]}%) outside realistic range"

            # Raw TSS and ESS should be positive
            assert metrics[n_factors]["raw_tss"][modality] > 0
            assert metrics[n_factors]["raw_ess"][modality] > 0

            # ESS should not exceed TSS
            assert (
                metrics[n_factors]["raw_ess"][modality]
                <= metrics[n_factors]["raw_tss"][modality]
            )

        # Total variance should be between modality-specific variances
        min_var = min(metrics[n_factors]["modality_variance"].values())
        max_var = max(metrics[n_factors]["modality_variance"].values())
        assert (
            min_var <= metrics[n_factors]["total_variance"] <= max_var
        ), f"Total variance ({metrics[n_factors]['total_variance']}%) outside range of modality variances ({min_var}% - {max_var}%)"

        # More factors should explain more variance
        if n_factors == 5:
            assert (
                metrics[n_factors]["total_variance"] >= metrics[3]["total_variance"]
            ), "More factors should explain more or equal variance"

            # Check the increase is not unrealistically large
            variance_increase = (
                metrics[n_factors]["total_variance"] - metrics[3]["total_variance"]
            )
            assert (
                variance_increase <= 30
            ), f"Unrealistic variance increase ({variance_increase}%) between 3 and 5 factors"


def test_regress_factors_with_formula(simple_mdata):
    """Test regression of MOFA factors against sample attributes."""
    # Test regression against group (categorical)
    group_results = mfa.regress_factors_with_formula(
        simple_mdata,
        formula="~ group",
        modality="rna",  # Use RNA modality for sample attributes
        progress_bar=False,
    )

    # Verify structure
    assert isinstance(group_results, pd.DataFrame)
    assert len(group_results) > 0
    assert all(
        col in group_results.columns
        for col in ["factor_name", "term", "estimate", "p_value", "q_value"]
    )

    # Factor 1 should be strongly related to group
    factor1_group = group_results[
        (group_results["factor_name"] == "Factor_1")
        & (
            group_results["term"] == "group[T.B]"
        )  # This is how statsmodels names categorical contrasts
    ]
    assert len(factor1_group) == 1
    assert factor1_group["p_value"].iloc[0] < 0.05  # Should be significant
    assert (
        abs(factor1_group["estimate"].iloc[0]) > 0.5
    )  # Should have strong effect size

    # Test regression against continuous variable
    cont_results = mfa.regress_factors_with_formula(
        simple_mdata, formula="~ continuous_var", modality="rna", progress_bar=False
    )

    # Factor 2 should be strongly related to continuous_var
    factor2_cont = cont_results[
        (cont_results["factor_name"] == "Factor_2")
        & (cont_results["term"] == "continuous_var")
    ]
    assert len(factor2_cont) == 1
    assert factor2_cont["p_value"].iloc[0] < 0.05  # Should be significant
    assert abs(factor2_cont["estimate"].iloc[0]) > 0.5  # Should have strong effect size

    # Test regression against both variables
    both_results = mfa.regress_factors_with_formula(
        simple_mdata,
        formula="~ group + continuous_var",
        modality="rna",
        progress_bar=False,
    )

    # Factor 3 should be related to both variables
    factor3_group = both_results[
        (both_results["factor_name"] == "Factor_3")
        & (both_results["term"] == "group[T.B]")
    ]
    factor3_cont = both_results[
        (both_results["factor_name"] == "Factor_3")
        & (both_results["term"] == "continuous_var")
    ]

    assert len(factor3_group) == 1
    assert len(factor3_cont) == 1
    assert factor3_group["p_value"].iloc[0] < 0.05  # Should be significant
    assert factor3_cont["p_value"].iloc[0] < 0.05  # Should be significant
    assert (
        abs(factor3_group["estimate"].iloc[0]) > 0.3
    )  # Should have moderate effect size
    assert (
        abs(factor3_cont["estimate"].iloc[0]) > 0.3
    )  # Should have moderate effect size

    # Factor 4 should not be strongly related to either variable
    factor4_both = both_results[both_results["factor_name"] == "Factor_4"]
    assert all(factor4_both["p_value"] > 0.05)  # Should not be significant
