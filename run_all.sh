#!/usr/bin/env bash
# ==============================================================================
# XPASS-Simple Automated Training Pipeline (run_all.sh)
#
# Automates sequential PIAA fine-tuning across 5 folds, 2 architectures (ICI, MIR),
# and 3 genres (art, fashion, scenery) with live ETA tracking, smart resume,
# and automated Google Drive synchronization.
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ─── Default Configurations ───────────────────────────────────────────────────
REMOTE="${RCLONE_REMOTE:-Google Drive}"
FOLDER_ID="${GDRIVE_FOLDER_ID:-1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L}"
DATASET_PREFIX="v4"
START_FOLD=1
ALL_FOLDS=(1 2 3 4 5)
ALL_MODELS=("ICI" "MIR")
ALL_GENRES=("art" "fashion" "scenery")
BATCH_SIZE=16
ROOT_DIR="Dataset"
FORCE=false
DRY_RUN=false
NO_UPLOAD=false

LOG_DIR="${SCRIPT_DIR}/logs_v4"
REPORTS_DIR="${SCRIPT_DIR}/reports"
MODELS_DIR="${SCRIPT_DIR}/models_pth"
MASTER_LOG="${SCRIPT_DIR}/run_all_master.log"

# ─── Help / Usage ─────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: ./run_all.sh [OPTIONS]

Options:
  --start-fold <N>       Start execution from fold N (1-5, default: 1)
  --folds <"1 2 3">      Specify custom list of folds (default: "1 2 3 4 5")
  --models <"ICI MIR">   Specify model architectures (default: "ICI MIR")
  --genres <"art ...">   Specify genres (default: "art fashion scenery")
  --batch-size <N>       Batch size for training (default: 16)
  --root-dir <path>      Dataset root directory (default: "Dataset")
  --remote <name>        rclone remote name (default: "Google Drive")
  --folder-id <id>       Google Drive folder ID (default: "1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L")
  --force                Force re-run even if job report already exists
  --dry-run              Display the execution plan without running training
  --no-upload            Disable automated Google Drive synchronization
  -h, --help             Show this help message and exit

Examples:
  ./run_all.sh
  ./run_all.sh --start-fold 2
  ./run_all.sh --models "ICI" --genres "art"
  ./run_all.sh --dry-run
EOF
  exit 0
}

# ─── Parse CLI Options ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-fold)
      START_FOLD="$2"
      shift 2
      ;;
    --folds)
      read -ra ALL_FOLDS <<< "$2"
      shift 2
      ;;
    --models)
      read -ra ALL_MODELS <<< "$2"
      shift 2
      ;;
    --genres)
      read -ra ALL_GENRES <<< "$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --root-dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --folder-id)
      FOLDER_ID="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift 1
      ;;
    --dry-run)
      DRY_RUN=true
      shift 1
      ;;
    --no-upload)
      NO_UPLOAD=true
      shift 1
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "❌ Unknown option: $1"
      usage
      ;;
  esac
done

# ─── Directories Setup ────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}" "${REPORTS_DIR}" "${MODELS_DIR}"

# ─── Signal Traps ─────────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "⚠️ [Pipeline Interrupted] Stopping child processes and exiting cleanly..."
  exit 130
}
trap cleanup SIGINT SIGTERM

# ─── Google Drive Sync Helper ─────────────────────────────────────────────────
sync_gdrive() {
  local src="$1"
  local dst="$2"
  local action="${3:-copy}"
  local extra="${4:-}"

  if [ "${NO_UPLOAD}" = true ] || [ "${DRY_RUN}" = true ]; then
    return 0
  fi

  echo "☁️ [GDrive Sync] ${action} ${src} -> ${REMOTE}:${dst}..."
  if [ -n "${extra}" ]; then
    rclone "${action}" "${src}" "${REMOTE}:${dst}" \
      --drive-root-folder-id "${FOLDER_ID}" \
      --retries 3 --low-level-retries 10 -q ${extra} || {
      echo "⚠️ [GDrive Warning] Sync failed for ${src}. Continuing pipeline..."
    }
  else
    rclone "${action}" "${src}" "${REMOTE}:${dst}" \
      --drive-root-folder-id "${FOLDER_ID}" \
      --retries 3 --low-level-retries 10 -q || {
      echo "⚠️ [GDrive Warning] Sync failed for ${src}. Continuing pipeline..."
    }
  fi
}

# ─── Build Job Queue ──────────────────────────────────────────────────────────
ACTIVE_FOLDS=()
for F in "${ALL_FOLDS[@]}"; do
  if [ "${F}" -ge "${START_FOLD}" ]; then
    ACTIVE_FOLDS+=("${F}")
  fi
done

TOTAL_JOBS=$((${#ACTIVE_FOLDS[@]} * ${#ALL_MODELS[@]} * ${#ALL_GENRES[@]}))

echo "======================================================================"
echo "🚀 XPASS-Simple Automation Pipeline"
echo "======================================================================"
echo "• Active Folds:        ${ACTIVE_FOLDS[*]} (start_fold=${START_FOLD})"
echo "• Architectures:       ${ALL_MODELS[*]}"
echo "• Genres:              ${ALL_GENRES[*]}"
echo "• Total Jobs:          ${TOTAL_JOBS}"
echo "• Batch Size:          ${BATCH_SIZE}"
echo "• Google Drive Remote: ${REMOTE}"
echo "• Target Folder ID:    ${FOLDER_ID}"
echo "• GDrive Sync:         $([ "${NO_UPLOAD}" = true ] && echo "Disabled" || echo "Enabled")"
echo "• Smart Resume:        $([ "${FORCE}" = true ] && echo "Disabled (Force Re-run)" || echo "Enabled")"
echo "======================================================================"
echo ""

if [ "${DRY_RUN}" = true ]; then
  echo "🔍 [Dry Run] Planned execution order:"
  JOB_COUNTER=0
  for F in "${ACTIVE_FOLDS[@]}"; do
    for M in "${ALL_MODELS[@]}"; do
      for G in "${ALL_GENRES[@]}"; do
        JOB_COUNTER=$((JOB_COUNTER + 1))
        echo "  [Job ${JOB_COUNTER}/${TOTAL_JOBS}] Fold: v4_fold${F} | Model: ${M} | Genre: ${G}"
      done
    done
  done
  echo ""
  echo "✅ Dry run complete. No training was executed."
  exit 0
fi

# ─── Execution Loop ───────────────────────────────────────────────────────────
START_TIMESTAMP=$(date +%s)
CURRENT_JOB=0
SUCCESS_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
FAILED_JOBS=()

for F in "${ACTIVE_FOLDS[@]}"; do
  echo ""
  echo "######################################################################"
  echo "  📁 STARTING FOLD: v4_fold${F}"
  echo "######################################################################"
  echo ""

  for M in "${ALL_MODELS[@]}"; do
    for G in "${ALL_GENRES[@]}"; do
      CURRENT_JOB=$((CURRENT_JOB + 1))
      DATASET_VER="${DATASET_PREFIX}_fold${F}"
      LOG_FILENAME="${M}_${G}_fold${F}.log"
      LOG_FILE="${LOG_DIR}/${LOG_FILENAME}"
      SAMPLES_ROOT="${ROOT_DIR}/sample/${G}_extracted"

      # Smart Resume Check: check if already completed
      if [ "${FORCE}" = false ] && [ -f "${LOG_FILE}" ]; then
        if grep -q "Evaluation Results" "${LOG_FILE}" 2>/dev/null || grep -q "Test Average" "${LOG_FILE}" 2>/dev/null; then
          echo "⏩ [Job ${CURRENT_JOB}/${TOTAL_JOBS}] Skipping already completed: ${DATASET_VER} | ${M} | ${G}"
          SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
          continue
        fi
      fi

      echo "----------------------------------------------------------------------"
      echo "▶️ [Job ${CURRENT_JOB}/${TOTAL_JOBS}] Running ${DATASET_VER} | ${M} | ${G}"
      echo "   • Log: ${LOG_FILE}"
      echo "   • Start: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "----------------------------------------------------------------------"

      JOB_START_TIME=$(date +%s)

      # Run training sequentially using uv
      if uv run python -m src.train_PIAA \
        --genre "${G}" \
        --dataset_ver "${DATASET_VER}" \
        --model_type "${M}" \
        --piaa_mode PIAA_finetune \
        --batch_size "${BATCH_SIZE}" \
        --root_dir "${ROOT_DIR}" \
        --samples_root "${SAMPLES_ROOT}" \
        --rclone_remote "${REMOTE}" \
        --gdrive_folder_id "${FOLDER_ID}" \
        $([ "${NO_UPLOAD}" = true ] && echo "--no_gdrive_upload") \
        2>&1 | tee "${LOG_FILE}"; then

        JOB_END_TIME=$(date +%s)
        JOB_DURATION=$((JOB_END_TIME - JOB_START_TIME))
        echo "✅ [Job ${CURRENT_JOB}/${TOTAL_JOBS}] Completed successfully in ${JOB_DURATION}s."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Sync job log and updated reports immediately
        sync_gdrive "${LOG_FILE}" "logs_v4/${LOG_FILENAME}" "copy"
        sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
      else
        echo "❌ [Job ${CURRENT_JOB}/${TOTAL_JOBS}] Failed with error (exit code $?)."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_JOBS+=("${DATASET_VER}_${M}_${G}")
      fi
    done
  done

  # ─── End-of-Fold Synchronization & Cleanup ─────────────────────────────────
  echo ""
  echo "📦 [Fold v4_fold${F} Completed] Synchronizing reports and logs to Google Drive..."
  sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
  sync_gdrive "${LOG_DIR}/" "logs_v4/" "copy"
  sync_gdrive "${MODELS_DIR}/" "models_pth/" "move" "--include *_finetune.pth"
done

# ─── Summary ──────────────────────────────────────────────────────────────────
END_TIMESTAMP=$(date +%s)
TOTAL_DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "======================================================================"
echo "🎉 PIPELINE EXECUTION SUMMARY"
echo "======================================================================"
echo "• Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "• Total Jobs:     ${TOTAL_JOBS}"
echo "• Succeeded:      ${SUCCESS_COUNT}"
echo "• Skipped:        ${SKIPPED_COUNT}"
echo "• Failed:         ${FAILED_COUNT}"

if [ "${FAILED_COUNT}" -gt 0 ]; then
  echo "⚠️ Failed Jobs:   ${FAILED_JOBS[*]}"
  exit 1
else
  echo "✅ All jobs completed successfully!"
  exit 0
fi
