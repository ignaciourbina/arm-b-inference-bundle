#!/usr/bin/env bash
# Fetch the raw Polis open-data for the 2014 Seattle $15/hour minimum-wage
# conversation into the path the reconstruction pipeline expects.
#
# Source: https://github.com/compdemocracy/openData/tree/master/15-per-hour-seattle
# (Computational Democracy Project open data.)
#
# Usage:  bash polis-analysis/fetch_polis_data.sh
# Then:   python polis-analysis/build_pipeline.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/datasets/polis-openData/15-per-hour-seattle"
BASE="https://raw.githubusercontent.com/compdemocracy/openData/master/15-per-hour-seattle"

# build_pipeline.py needs comments.csv + participants-votes.csv; the rest are
# fetched for completeness/provenance.
FILES=(comments.csv participants-votes.csv votes.csv summary.csv stats-history.csv)

mkdir -p "$DEST"
echo "Fetching Polis 15-per-hour-seattle data -> $DEST"
for f in "${FILES[@]}"; do
  echo "  - $f"
  curl -fsSL "$BASE/$f" -o "$DEST/$f"
done

echo "Done. Required files:"
ls -la "$DEST/comments.csv" "$DEST/participants-votes.csv"
echo
echo "Next: python polis-analysis/build_pipeline.py"
