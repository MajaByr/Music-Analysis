#!/bin/bash

# Set pipefail in case of errors
set -eo pipefail

echo "Setting up Music-Analysis repository..."

# Remove old python environment
rm -rf .venv/

# Create python environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip3 install -r requirements.txt

echo "===== ENVIRONMENT SET UP SUCCESFULLY ====="