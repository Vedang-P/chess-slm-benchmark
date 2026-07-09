#!/bin/bash
set -euo pipefail

# Download Lost in Aggregation maze datasets from GitHub Releases v0.1
# https://github.com/s-nlp/lost-in-aggregation/releases/tag/v0.1

DATA_DIR="/home/vedang/Desktop/Research/neuro-symbolic-pathfinding/data/lost_in_aggregation"
mkdir -p "$DATA_DIR"

BASE_URL="https://github.com/s-nlp/lost-in-aggregation/releases/download/v0.1"
FILES=(
    "size_3_150.json"
    "size_5_150.json"
    "size_7_150.json"
    "size_10_150.json"
    "size_15_150.json"
    "size_20_150.json"
    "size_30_150.json"
)

echo "Downloading Lost in Aggregation mazes..."
for f in "${FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "  Downloading $f..."
        wget -q --show-progress "$BASE_URL/$f" -O "$DATA_DIR/$f"
    else
        echo "  $f already exists, skipping"
    fi
done

echo "Verifying files..."
for f in "${FILES[@]}"; do
    if [ -f "$DATA_DIR/$f" ]; then
        SIZE=$(stat -c%s "$DATA_DIR/$f" 2>/dev/null || stat -f%z "$DATA_DIR/$f" 2>/dev/null)
        echo "  $f: $SIZE bytes"
    else
        echo "  MISSING: $f"
    fi
done

echo "Done. Files in $DATA_DIR"
ls -la "$DATA_DIR"
