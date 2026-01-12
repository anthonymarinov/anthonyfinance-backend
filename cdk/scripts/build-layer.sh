#!/bin/bash
set -e

# Extract dependencies from pyproject.toml
python3 -c "import tomllib; deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(deps))" > /tmp/deps.txt

# Install dependencies to Lambda layer path
pip install --no-cache-dir --target /asset-output/python -r /tmp/deps.txt

# Cleanup unnecessary files
cd /asset-output/python
rm -rf */tests */.pytest_cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
