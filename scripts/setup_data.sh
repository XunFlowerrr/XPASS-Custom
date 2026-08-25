#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_data.sh
# Sets up Python environment, downloads/extracts datasets, and downloads
# pretrained model weights for XPASS baseline v4.
# ==============================================================================

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SKIP_MODELS=false
KEEP_ZIPS=false
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=true ;;   # clone already carries models_pth/
    --keep-zips)   KEEP_ZIPS=true ;;     # keep archives after extraction
    -h|--help)
      echo "Usage: scripts/setup_data.sh [--skip-models] [--keep-zips]"
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# --- Google Drive File IDs (Public Share) -------------------------------------
SAMPLE_ID="18EMET1QDfQgVrJeC1VWbuUK0U7b7cuWT"      # sample.zip ~1.7G (images for art, fashion, scenery)
ESSENTIALS_ID="1xsGRhDSWavs6ySVeWfcFUcobgGuiS5Vb"  # xpass_v4_essentials.zip ~6.4M (Dataset/ CSV & splits)
ART_MODELS_ID="1TlA5OAoF4cKNnvNjTlCt9Nt41jBT3_et"      # art_models_v4.zip (pretrained weights for art)
FASHION_MODELS_ID="1G99y0g42h8Hfbk6Ew_85zUYNKPYuaDiW"  # fashion_models_v4.zip (pretrained weights for fashion)
SCENERY_MODELS_ID="1raHHwn5qDdG_id4r0NaE24uz6pN47s88"  # scenery_models_v4.zip (pretrained weights for scenery)

# ─── Download Validation ──────────────────────────────────────────────────────
# A non-empty file is not a valid archive. When Google Drive rate-limits a public
# file it answers with a few KB of HTML, which lands under the expected name and
# passes any "is it non-empty" test -- the failure then surfaces much later as a
# confusing unzip error. Check the size floor and the archive itself instead.
verify_zip() {
  local f="$1" min_bytes="$2" size
  [ -f "$f" ] || return 1
  size=$(wc -c < "$f" | tr -d ' ')
  if [ "$size" -lt "$min_bytes" ]; then
    echo "   ⚠️ $f is $size bytes, expected at least $min_bytes" >&2
    return 1
  fi
  unzip -t "$f" >/dev/null 2>&1
}

# ─── Robust Download Function ─────────────────────────────────────────────────
dl() {
  local id="$1" out="$2" min_bytes="${3:-1000000}"
  if verify_zip "$out" "$min_bytes"; then
    echo ">> $out already present and valid, skipping download."
    return 0
  fi
  if [ -f "$out" ]; then
    echo ">> $out is incomplete or corrupt, re-downloading."
    rm -f "$out"
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

  if ! verify_zip "$out" "$min_bytes"; then
    rm -f "$out"
    echo "❌ ERROR: $out did not download as a valid archive." >&2
    echo "   Google Drive rate-limits heavily downloaded public files and answers" >&2
    echo "   with an HTML notice instead of the data. Retry later, or fetch the" >&2
    echo "   file by hand and drop it in ${REPO_ROOT}." >&2
    exit 1
  fi
}

echo "=================================================="
echo " [1/5] Sync Python Environment (uv)"
echo "=================================================="
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ ERROR: 'uv' is required. Please install uv (https://docs.astral.sh/uv/)" >&2
  exit 1
fi
uv sync

echo "=================================================="
echo " [2/5] Download Datasets & Pretrained Models"
echo "=================================================="
dl "$SAMPLE_ID" sample.zip 1500000000
dl "$ESSENTIALS_ID" xpass_v4_essentials.zip 5000000
if [ "$SKIP_MODELS" = true ]; then
  echo ">> --skip-models: not downloading pretrained weights."
else
  dl "$ART_MODELS_ID" art_models_v4.zip 4000000000
  dl "$FASHION_MODELS_ID" fashion_models_v4.zip 4000000000
  dl "$SCENERY_MODELS_ID" scenery_models_v4.zip 4000000000
fi

echo "=================================================="
echo " [3/5] Extract and Organize Datasets"
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

# Clean up nested zip files to save disk space
rm -f Dataset/sample/art.zip Dataset/sample/fashion.zip Dataset/sample/scenery_image.zip sample.zip.tmp* xpass_v4_essentials.zip.tmp*
find Dataset -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "=================================================="
echo " [4/5] Extract Pretrained Model Weights"
echo "=================================================="
mkdir -p models_pth

if [ "$SKIP_MODELS" = true ]; then
  echo ">> --skip-models: keeping the existing models_pth/."
else
  for mzip in art_models_v4.zip fashion_models_v4.zip scenery_models_v4.zip; do
    if [ -f "$mzip" ]; then
      echo ">> Extracting $mzip..."
      unzip -o -q "$mzip" -d models_pth/ || unzip -o -q "$mzip"
    fi
  done
fi

# Clean up nested directory structures if zip contained models_pth/
if [ -d "models_pth/models_pth" ]; then
  cp -r models_pth/models_pth/* models_pth/ 2>/dev/null || true
  rm -rf models_pth/models_pth
fi
find models_pth -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# The archives are 16.7 GB together and are pure duplication once extracted.
# Leaving them behind doubles what a studio snapshot has to carry.
if [ "$KEEP_ZIPS" = false ]; then
  echo ">> Removing extracted archives to reclaim disk (--keep-zips to retain)..."
  rm -f sample.zip xpass_v4_essentials.zip \
        art_models_v4.zip fashion_models_v4.zip scenery_models_v4.zip
fi

echo "=================================================="
echo " [5/5] Verify Dataset & Pretrained Models"
echo "=================================================="
ok=1

# Verify images
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

# Verify splits
for f in Dataset/maked/users.csv Dataset/split/v4_fold1/art/train_PIAA.txt; do
  if [ -f "$f" ]; then
    printf "  ✅ Found essential file: %s\n" "$f"
  else
    printf "  ❌ Missing essential file: %s\n" "$f"
    ok=0
  fi
done

# Verify model weights
for g in art fashion scenery; do
  dir="models_pth/v4_fold1/${g}"
  cnt=$(find "${dir}" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$cnt" -gt 0 ]; then
    printf "  ✅ %-8s %2s pretrained weights in %s\n" "$g" "$cnt" "$dir"
  else
    printf "  ❌ %-8s No pretrained weights found in %s\n" "$g" "$dir"
    ok=0
  fi
done

if [ "$ok" -eq 1 ]; then
  echo ""
  echo "🎉 Environment, datasets, and pretrained weights are completely verified and ready for training!"
else
  echo ""
  echo "❌ Some required files are missing. Please check the logs above." >&2
  exit 1
fi
