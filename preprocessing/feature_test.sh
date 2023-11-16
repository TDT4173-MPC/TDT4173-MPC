#!/bin/bash

# Directory containing the test plan files
test_plan_dir="preprocessing/test_plan"

# Loop through each file in the test plan directory
for file in "$test_plan_dir"/*; do
  # Check if the item is a file
  if [ -f "$file" ]; then
    # Extract the filename without the path
    filename=$(basename "$file")
    
    # Run the preprocessing script with the corresponding file in the columns directory
    preprocessing/preprocessing.sh "preprocessing/test_plan/$filename"
  fi
done
