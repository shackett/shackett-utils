import os
import pytest
import tempfile

import muon
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
            seed=42
        )
        
        # Calculate variance metrics
        metrics = mfa._calculate_mofa_variance_metrics(simple_mdata, use_layer="log2_centered")
        
        # Basic validation of metrics structure
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == {"total_variance", "modality_variance", "raw_tss", "raw_ess"}
        assert set(metrics["modality_variance"].keys()) == {"rna", "atac"}
        
        # Validate variance calculations
        for modality in ["rna", "atac"]:
            # Variance explained should be between 0 and 100%
            assert 0 <= metrics["modality_variance"][modality] <= 100, \
                f"Variance for {modality} ({metrics['modality_variance'][modality]}%) outside valid range"
            
            # Raw TSS and ESS should be positive
            assert metrics["raw_tss"][modality] > 0
            assert metrics["raw_ess"][modality] > 0
            
            # ESS should not exceed TSS
            assert metrics["raw_ess"][modality] <= metrics["raw_tss"][modality] * 1.01  # Allow 1% numerical error
        
        # Total variance should be weighted average of modality variances
        total_tss = sum(metrics["raw_tss"].values())
        weights = {mod: tss / total_tss for mod, tss in metrics["raw_tss"].items()}
        expected_total = sum(
            metrics["modality_variance"][mod] * weights[mod]
            for mod in metrics["modality_variance"].keys()
        )
        np.testing.assert_allclose(metrics["total_variance"], expected_total, rtol=1e-10)


def test_mofa_factor_analysis(simple_mdata, tmp_path):
    """Test MOFA factor analysis and variance calculations."""
    # Set up temporary directory for MOFA results
    models_dir = os.path.join(tmp_path, "mofa_models")
    
    # Run MOFA with a small number of factors
    factor_results = mfa.run_mofa_factor_scan(
        simple_mdata,
        factor_range=[2, 4],  # Test with just 2 factor values for speed
        use_layer="log2_centered",
        use_var=None,  # Explicitly disable variable selection
        models_dir=models_dir,
        seed=42
    )
    
    # Basic validation of results structure
    assert set(factor_results.keys()) == {2, 4}
    for n_factors in [2, 4]:
        assert "model_file" in factor_results[n_factors]
        assert os.path.exists(factor_results[n_factors]["model_file"])
    
    # Calculate variance metrics using calculate_variance_metrics
    metrics_from_scan = mfa.calculate_variance_metrics(
        factor_results,
        mdata=simple_mdata,
        use_layer="log2_centered"
    )
    
    # For each factor value, also calculate metrics directly and compare
    for n_factors in [2, 4]:
        # Load model into a fresh copy of the data
        test_data = simple_mdata.copy()
        mfa._mofa(test_data, outfile=factor_results[n_factors]["model_file"], use_var=None)  # Explicitly disable variable selection
        
        # Calculate metrics directly
        direct_metrics = mfa._calculate_mofa_variance_metrics(test_data, use_layer="log2_centered")
        
        # Validate metric structure
        assert isinstance(direct_metrics, dict)
        assert set(direct_metrics.keys()) == {"total_variance", "modality_variance", "raw_tss", "raw_ess"}
        assert set(direct_metrics["modality_variance"].keys()) == {"rna", "atac"}
        
        # Compare direct calculation with factor scan results
        scan_metrics = metrics_from_scan[n_factors]
        np.testing.assert_allclose(
            direct_metrics["total_variance"],
            scan_metrics["total_variance"],
            rtol=1e-10
        )
        
        # Validate variance calculations
        for modality in ["rna", "atac"]:
            # Variance explained should be between 0 and 100%
            assert 0 <= direct_metrics["modality_variance"][modality] <= 100, \
                f"Variance for {modality} ({direct_metrics['modality_variance'][modality]}%) outside valid range"
            
            # Raw TSS and ESS should be positive
            assert direct_metrics["raw_tss"][modality] > 0
            assert direct_metrics["raw_ess"][modality] > 0
            
            # ESS should not exceed TSS
            assert direct_metrics["raw_ess"][modality] <= direct_metrics["raw_tss"][modality] * 1.01  # Allow 1% numerical error
            
            # Compare modality variances between direct and scan calculations
            np.testing.assert_allclose(
                direct_metrics["modality_variance"][modality],
                scan_metrics["modality_variance"][modality],
                rtol=1e-10
            )
        
        # Total variance should be weighted average of modality variances
        total_tss = sum(direct_metrics["raw_tss"].values())
        weights = {mod: tss / total_tss for mod, tss in direct_metrics["raw_tss"].items()}
        expected_total = sum(
            direct_metrics["modality_variance"][mod] * weights[mod]
            for mod in direct_metrics["modality_variance"].keys()
        )
        np.testing.assert_allclose(direct_metrics["total_variance"], expected_total, rtol=1e-10) 