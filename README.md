# Shackett Utils

Personal utility functions for data analysis, currently focused on multi-omic genomics workflows.

## Overview

This is my personal collection of analysis utilities, extracted from various blog posts and research projects. The current focus is heavily on **multi-modal genomics analysis** (transcriptomics, proteomics, etc.) using the scverse ecosystem, but the modular structure allows me to add new domains as I explore different problems.

**Current capabilities include:**
- Multi-omic factor analysis with variance optimization and interpretation
- Feature-wise regression analysis with batch effect correction  
- Exploratory data analysis tools for AnnData/MuData objects
- Statistical utilities with proper multiple testing correction
- Dataset-specific functions (currently: clinical data imputation)

The package emphasizes working with **AnnData/MuData** data structures and producing **tidy, analysis-ready outputs**.

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/shackett/shackett-utils.git

# Install with extras for specific use cases
pip install "git+https://github.com/shackett/shackett-utils.git[genomics]"      # AnnData/MuData + scverse tools
pip install "git+https://github.com/shackett/shackett-utils.git[statistics]"   # Advanced statistical modeling  
pip install "git+https://github.com/shackett/shackett-utils.git[viz]"          # Visualization dependencies
pip install "git+https://github.com/shackett/shackett-utils.git[all]"          # Everything

# Development installation
git clone https://github.com/shackett/shackett-utils
cd shackett-utils
pip install -e .[all,test]
```

## Quick Examples

### Multi-omic Factor Analysis
```python
from shackett_utils.genomics import mdata_factor_analysis as mfa

# Optimize number of factors across a range
results = mfa.run_mofa_factor_scan(mdata, factor_range=range(5, 31, 5))

# Visualize results and select optimal number
figures = mfa.visualize_factor_scan_results(results, user_factors=15)

# Test factor associations with metadata
associations = mfa.regress_factors_with_formula(mdata, formula="~ treatment + batch")
```

### Regression Analysis with Batch Correction
```python
from shackett_utils.genomics import adata_regression

# Fit models with smooth batch effects
results = adata_regression.adata_model_fitting(
    adata, 
    formula="~ treatment + s(batch_date)",  # GAM with spline
    n_jobs=4
)

# Store results in the AnnData object
adata_regression.add_regression_results_to_anndata(adata, results, inplace=True)
```

### Exploratory Data Analysis
```python
from shackett_utils.genomics import mdata_eda

# Correlate PCs with sample metadata across modalities
results = mdata_eda.analyze_pc_metadata_correlation_mudata(
    mdata, prioritized_vars=["treatment", "batch"]
)
```

## Package Structure

The modular design reflects how I actually work - different projects need different tools:

- **`genomics`** - Multi-modal analysis workflows (AnnData/MuData focused)
- **`statistics`** - Statistical modeling and hypothesis testing utilities
- **`applications`** - Dataset-specific functions (e.g., `forny_imputation` for clinical data)
- **`utils`** - General utilities and helper functions

Each module has optional dependencies, so you only install what you need.

## Philosophy 

This package grew out of repeatedly writing similar analysis code across projects. Rather than a comprehensive framework, it's a collection of **opinionated utilities** that:

- Work well with the **scverse ecosystem** (scanpy, muon, etc.)
- Produce **tidy outputs** that integrate easily with downstream analysis
- Handle common statistical challenges (batch effects, missing data, multiple testing)
- Emphasize **reproducible workflows** over one-off scripts

The focus on multi-omic genomics reflects my current research interests, but the modular structure means I can easily add new domains as my work evolves.

## Development

```bash
git clone https://github.com/shackett/shackett-utils
cd shackett-utils
pip install -e .[all,test]
pytest
```

## Usage Note

This is primarily designed for my own use, so the API may change as my needs evolve. That said, the core genomics functionality is fairly stable since it's been extracted from multiple completed analyses. Feel free to use or adapt anything that's helpful for your own work.

## License

MIT License - see [LICENSE](LICENSE) file for details.