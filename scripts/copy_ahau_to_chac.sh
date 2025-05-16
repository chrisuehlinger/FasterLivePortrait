#!/bin/bash
# Copy landmarks from ahau-kin.fsp (human) to chac-bolay.fsp (animal)
# This script demonstrates how to use the copy_landmarks.py CLI

# Set paths - modify these if your files are in different locations
SOURCE_FSP="assets/examples/source/chac-bolay_with_ahau_landmarks.fsp"
TARGET_FSP="assets/examples/source/chac-bolay.fsp"
OUTPUT_FSP="assets/examples/source/chac-bolay.fsp"

# Ensure the script has proper permissions
chmod +x scripts/copy_landmarks.py

# Run the landmark copying CLI
python3 scripts/copy_landmarks.py \
  --source "$SOURCE_FSP" \
  --target "$TARGET_FSP" \
  --output "$OUTPUT_FSP"

echo "Landmark copying complete!"
echo "Source: $SOURCE_FSP"
echo "Target: $TARGET_FSP"
echo "Output: $OUTPUT_FSP"

# Optional: Display a visualization of the result
echo "To visualize the result, run:"
echo "python3 scripts/fsp_viewer.py $OUTPUT_FSP"

python3 scripts/fsp_viewer.py $OUTPUT_FSP
