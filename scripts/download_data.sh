#!/usr/bin/env bash
# Download evaluation data into data/ (gitignored). Resumable; safe to re-run.
#   COCO val2017 images + 2017 annotations  (~1.07 GB)
#   ADE20K SceneParsing (ADEChallengeData2016, ~0.97 GB; val = 2000 images)
# SHA-256 of every archive is written to configs/data_sha256.txt (committed) so a later
# download can be checked against the one used for the results.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW" data/coco data/ade20k

fetch() {  # url dest  (parallel ranged download; hosts throttle per connection)
  echo ">> $1"
  python3 -I scripts/fetch.py "$1" "$2" --workers 16
}
extract() {  # zip dest_dir marker
  if [[ ! -e "$3" ]]; then
    echo ">> extracting $1"
    python3 -I -m zipfile -e "$1" "$2"
  fi
}

fetch http://images.cocodataset.org/zips/val2017.zip "$RAW/val2017.zip"
fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$RAW/annotations_trainval2017.zip"
# https: plain http on this host measured ~0.3 kB/s vs ~157 kB/s/connection over https
fetch https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip "$RAW/ADEChallengeData2016.zip"

extract "$RAW/val2017.zip" data/coco data/coco/val2017
extract "$RAW/annotations_trainval2017.zip" data/coco data/coco/annotations
extract "$RAW/ADEChallengeData2016.zip" data/ade20k data/ade20k/ADEChallengeData2016

(cd "$RAW" && sha256sum val2017.zip annotations_trainval2017.zip ADEChallengeData2016.zip) \
  > configs/data_sha256.txt

n_coco=$(find data/coco/val2017 -name '*.jpg' | wc -l)
n_ade=$(find data/ade20k/ADEChallengeData2016/images/validation -name '*.jpg' | wc -l)
echo ">> COCO val2017 images: $n_coco (expected 5000)"
echo ">> ADE20K val images:   $n_ade (expected 2000)"
[[ "$n_coco" == 5000 && "$n_ade" == 2000 ]]
