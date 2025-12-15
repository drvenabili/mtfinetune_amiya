#!/bin/bash

# Directory to clean (default: current directory)
TARGET_DIR="${1:-.}"

echo "This will delete all .error and .out files in: $TARGET_DIR"
echo
read -p "Are you sure you want to continue? (y/N): " confirm

# Convert to lowercase
confirm=$(echo "$confirm" | tr 'A-Z' 'a-z')

if [[ "$confirm" == "y" || "$confirm" == "yes" ]]; then
    echo "Deleting files..."
    find "$TARGET_DIR" -maxdepth 1 -type f \( -name "*.error" -o -name "*.out" \) -print -delete
    echo "Done."
else
    echo "Aborted. No files were deleted."
fi
