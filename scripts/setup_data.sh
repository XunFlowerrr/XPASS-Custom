#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_data.sh
# Sets up environment and downloads/extracts datasets for XPASS baseline v4.
# ==============================================================================

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Google Drive file IDs (public share) ---------------------------------------
SAMPLE_ID="18EMET1QDfQgVrJeC1VWbuUK0U7b7cuWT"      # sample.zip ~1.7G (images for art, fashion, scenery)
ESSENTIALS_ID="1xsGRhDSWavs6ySVeWfcFUcobgGuiS5Vb"  # xpass_v4_essentials.zip ~6.4M (Dataset/ CSV & splits)

dl() {
  local id="$1" out="$2"
  if [ -s "$out" ]; then
    echo ">> $out already exists, skipping download."
    return 0
  fi

  echo ">> Downloading $out (Google Drive ID: $id)..."
  if command -v gdown >/dev/null 2>&1; then
    gdown "$id" -O "$out"
  elif command -v uv >/dev/null 2>&1; then
    uvx gdown "$id" -O "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -c --no-verbose \
      "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" \
      -O "$out"
  else
    curl -L -C - \
      "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" \
      -o "$out"
  fi

  if [ ! -s "$out" ]; then
    echo "❌ ERROR: Failed to download $out. Please check your internet connection." >&2
    exit 1
  fi
}

echo "=================================================="
echo " [1/4] Sync Python Environment (uv)"
echo "=================================================="
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ ERROR: 'uv' is required. Please install uv (https://docs.astral.sh/uv/)" >&2
  exit 1
fi
uv sync

echo "=================================================="
echo " [2/4] Download Datasets"
echo "=================================================="
dl "$SAMPLE_ID" sample.zip
dl "$ESSENTIALS_ID" xpass_v4_essentials.zip

echo "=================================================="
echo " [3/4] Extract and Organize Datasets"
echo "=================================================="
# Extract Dataset/ essentials (CSV + split)
echo ">> Extracting xpass_v4_essentials.zip..."
unzip -o -q xpass_v4_essentials.zip

# Ensure Dataset directory exists
mkdir -p Dataset/sample

# Extract sample.zip if not already extracted
if [ ! -d "Dataset/sample/art_extracted" ]; then
  echo ">> Extracting sample.zip into Dataset/sample..."
  if [ -d "sample" ]; then
    rm -rf sample
  fi
  unzip -q sample.zip
  # Move extracted sample directories into Dataset/sample/
  if [ -d "sample" ]; then
    cp -r sample/* Dataset/sample/ 2>/dev/null || mv sample/* Dataset/sample/ 2>/dev/null || true
    rm -rf sample
  fi
fi

# Clean up nested zip files (e.g. art.zip) to save ~850MB
rm -f Dataset/sample/art.zip Dataset/sample/fashion.zip Dataset/sample/scenery_image.zip sample.zip.tmp* xpass_v4_essentials.zip.tmp*
find Dataset -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "=================================================="
echo " [4/4] Verify Dataset Integrity"
echo "=================================================="
ok=1
for pair in "art:Dataset/sample/art_extracted/art" \
            "fashion:Dataset/sample/fashion_extracted/fashion" \
            "scenery:Dataset/sample/scenery_extracted/scenery_image"; do
  g="${pair%%:*}"
  d="${pair##*:}"
  n=$(find "$d" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then
    printf "  ✅ %-8s %5s images in %s\n" "$g" "$n" "$d"
  else
    printf "  ❌ %-8s No images found in %s\n" "$g" "$d"
    ok=0
  fi
done

for f in Dataset/maked/users.csv Dataset/split/v4_fold1/art/train_PIAA.txt; do
  if [ -f "$f" ]; then
    printf "  ✅ Found essential file: %s\n" "$f"
  else
    printf "  ❌ Missing essential file: %s\n" "$f"
    ok=0
  fi
done

if [ "$ok" -eq 1 ]; then
  echo ""
  echo "🎉 Dataset setup completed successfully! Ready for training."
else
  echo ""
  echo "❌ Some dataset files are missing. Please check the logs above." >&2
  exit 1
fi
