#!/usr/bin/env bash
# Go-live (or roll back) for the interactions data (DECISIONS D-055): swap the experiment build into the production
# folders that `make serve` / `make deploy` use. Run with the branch that sets config/pipeline.yaml to the
# interactions data checked out. Nothing is deployed; `make deploy` is a separate step.
#
#   scripts/golive_interactions.sh          artifacts_interactions/ -> artifacts/, data/interactions/ -> data/interim/;
#                                           today's build kept in artifacts_reviews/ and data/archive/reviews/
#   scripts/golive_interactions.sh --rollback   the reverse
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--rollback" ]; then
  [ -d artifacts_reviews ] && [ -d data/archive/reviews ] || { echo "rollback: no archived reviews build"; exit 1; }
  mv artifacts artifacts_interactions
  mv artifacts_reviews artifacts
  mv data/interim data/interactions
  mv data/archive/reviews data/interim
  echo "rolled back: artifacts/ and data/interim/ are the reviews build again (check out main's config too)"
  exit 0
fi

[ -f artifacts_interactions/manifest.json ] || { echo "go-live: no experiment build in artifacts_interactions/"; exit 1; }
[ ! -e artifacts_reviews ] && [ ! -e data/archive/reviews ] || { echo "go-live: an archived build already exists"; exit 1; }
grep -q "ratings: goodreads_interactions_dedup" config/pipeline.yaml || { echo "go-live: config/pipeline.yaml isn't on the interactions data"; exit 1; }

# The experiment linked the ratings-independent book metadata from data/interim; make those real files first.
for f in data/interactions/*; do
  if [ -L "$f" ]; then target=$(readlink "$f"); rm "$f"; cp "$target" "$f"; fi
done
mkdir -p data/archive
mv data/interim data/archive/reviews
mv data/interactions data/interim
mv artifacts artifacts_reviews
mv artifacts_interactions artifacts
python3 -c "import json; m = json.load(open('artifacts/manifest.json')); print('go-live: artifacts/ is now', m.get('counts'))"
echo "go-live: done. Reviews build kept in artifacts_reviews/ and data/archive/reviews/ (rollback: --rollback)."
