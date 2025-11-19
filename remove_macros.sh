#!/bin/bash

# This script removes lines containing a specific warning message ("...have a way to export this macro.")
# from all .md files within a specified directory and its subdirectories.
# It uses sed for in-place editing.

# --- Validation ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_path>" >&2
    exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a valid directory." >&2
    exit 1
fi

# --- Main Logic ---

echo "Starting macro and details tag removal in '$TARGET_DIR'..."

# Find all files ending in .md and execute sed on them.
# - `find ... -print0` and `xargs -0 ...` are used to correctly handle
#   filenames that may contain spaces or other special characters.
# - `sed -i '' ...` performs the edit in-place without creating a backup file
#   on macOS/BSD.
# - The '/pattern/d' command deletes the matching line.
find "$TARGET_DIR" -type f -name "*.md" -print0 | xargs -0 sed -i '' "/have a way to export this macro/d"
find "$TARGET_DIR" -type f -name "*.md" -print0 | xargs -0 sed -i '' "/<details>/d"
find "$TARGET_DIR" -type f -name "*.md" -print0 | xargs -0 sed -i '' "/<\/details>/d"
find "$TARGET_DIR" -type f -name "*.md" -print0 | xargs -0 sed -i '' "/<summary>Click here to expand...<\/summary>/d"

echo "Done. All relevant files have been processed."
