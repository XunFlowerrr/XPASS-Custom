#!/usr/bin/env bash
# ==============================================================================
# scripts/run_fold.sh <fold> [jobs] [num_workers]
#
# Runs one fold's finetune sweep and leaves a sentinel behind. Meant to be
# launched detached, so the run outlives the SSH/SDK session that started it:
#
#   nohup setsid bash scripts/run_fold.sh 2 2 4 </dev/null >/dev/null 2>&1 &
#
# The final rclone flush happens BEFORE the sentinel is written. The orchestrator
# stops a machine the moment it sees the sentinel, so writing it first would let
# a stop truncate an upload.
# ==============================================================================
set -uo pipefail

FOLD="${1:?usage: run_fold.sh <fold> [jobs] [num_workers]}"
JOBS="${2:-2}"
NUM_WORKERS="${3:-}"

REPO_DIR="${REPO_DIR:-$HOME/XPASS-Custom}"
REMOTE_NAME="${RCLONE_REMOTE:-Google Drive}"
FOLDER_ID="${GDRIVE_FOLDER_ID:-1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L}"

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
STATUS_DIR="$HOME/fleet/status"
FOLD_LOG="$HOME/fleet/fold${FOLD}.log"
mkdir -p "$STATUS_DIR" "$(dirname "$FOLD_LOG")"

cd "$REPO_DIR" || { echo "❌ no repo at $REPO_DIR" >&2; echo 127 > "${STATUS_DIR}/fold${FOLD}.done"; exit 127; }

# Clear any sentinel from a previous attempt so a retry cannot be read as done.
rm -f "${STATUS_DIR}/fold${FOLD}.done"
date +%s > "${STATUS_DIR}/fold${FOLD}.started"
echo "$$" > "${STATUS_DIR}/fold${FOLD}.pid"

{
  echo "=== fold ${FOLD} | jobs=${JOBS} | workers=${NUM_WORKERS:-auto} ==="
  echo "=== commit $(git rev-parse --short HEAD) | $(date -Is) ==="
} > "$FOLD_LOG"

./run_all.sh \
  --stage finetune \
  --folds "${FOLD}" \
  --jobs "${JOBS}" \
  ${NUM_WORKERS:+--num-workers "${NUM_WORKERS}"} \
  --no-agg \
  --skip-setup \
  >> "$FOLD_LOG" 2>&1
RC=$?

# ─── Final flush, before the sentinel ─────────────────────────────────────────
echo "=== flushing results to Drive ===" >> "$FOLD_LOG"
for pair in "reports/:reports/" "logs_v4/:logs_v4/"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  [ -d "$src" ] || continue
  rclone copy "$src" "${REMOTE_NAME}:${dst}" \
    --drive-root-folder-id "$FOLDER_ID" \
    --retries 5 --low-level-retries 10 -q >> "$FOLD_LOG" 2>&1 \
    || echo "⚠️ flush failed for ${src}; local copy retained" >> "$FOLD_LOG"
done
rclone copy "$FOLD_LOG" "${REMOTE_NAME}:logs_v4/" \
  --drive-root-folder-id "$FOLDER_ID" --retries 3 -q 2>/dev/null || true

date +%s > "${STATUS_DIR}/fold${FOLD}.finished"
echo "$RC" > "${STATUS_DIR}/fold${FOLD}.done"
echo "=== fold ${FOLD} finished rc=${RC} ===" >> "$FOLD_LOG"
exit "$RC"
