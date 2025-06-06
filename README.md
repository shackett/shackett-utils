# Shackett Utils

Personal utility functions for blog posts and applications.

## Installation

```bash
# Basic installation
pip install shackett-utils

# With optional dependencies
pip install shackett-utils[genomics]        # Genomics-oriented
pip install shackett-utils[statistics]      # Statistical analysis 
pip install shackett-utils[all]        # Everything

# Development installation
pip install -e .[test]
```

## Quick Start

```python
import shackett_utils

# Common functions available at top level
text = shackett_utils.clean_text("  messy   text  ")
content = shackett_utils.safe_read_file("data.txt")
duration = shackett_utils.pretty_duration(3661)  # "1.0h"

# Subpackage functionality
links = shackett_utils.web.extract_links(html_content)
df = shackett_utils.data.smart_read_csv("data.csv")
```

## Subpackages

- `genomics` - Functions relating to common genomic `scverse` data structures 
- `statistics` - 
- `eda` - File system operations
- `misc` - Random utility functions

## Development

```bash
git clone https://github.com/shackett/shackett-utils
cd shackett-utils
pip install -e .[all,test]
pytest
```