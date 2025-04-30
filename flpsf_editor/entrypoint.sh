#!/bin/bash

# Make script exit when any command fails
set -e

# Check if any arguments are passed
if [ -z "$1" ]; then
  # No arguments, start with default configuration
  echo "Starting FLPSF Editor with default configuration..."
  python main.py
else
  # Arguments provided, pass them to main.py
  echo "Starting FLPSF Editor with custom arguments..."
  python main.py "$@"
fi