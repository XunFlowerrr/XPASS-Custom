#!/usr/bin/env bash
# ==============================================================================
# XPASS-Simple Automated End-to-End Pipeline (run_all.sh)
#
# Automates the complete 4-stage pipeline:
#   [1/4] GIAA Training (NIMA backbone per fold & genre)
#   [2/4] PIAA Pretrain (ICI & MIR pretrain per fold & genre)
#   [3/4] PIAA Finetune (Personalized user models & evaluation)
#   [4/4] Aggregate Metrics (Calculates CCC / SROCC per genre)
#
# Features:
#   • Auto-setup datasets if missing (scripts/setup_data.sh)
#   • Sequential GPU/MPS execution for stability and memory safety
#   • Smart resume (skips completed stages/jobs automatically)
#   • Real-time Google Drive synchronization via rclone
#   • Process safety traps (clean cancellation on SIGINT/SIGTERM)
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ─── Default Configurations ───────────────────────────────────────────────────
REMOTE="${RCLONE_REMOTE:-Google Drive}"
FOLDER_ID="${GDRIVE_FOLDER_ID:-1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L}"
DATASET_PREFIX="v4"
STAGE="finetune"
START_FOLD=1
ALL_FOLDS=(1 2 3 4 5)
ALL_MODELS=("ICI" "MIR")
ALL_GENRES=("art" "fashion" "scenery")
BATCH_SIZE=16
ROOT_DIR="Dataset"
FORCE=false
DRY_RUN=false
NO_UPLOAD=false
KEEP_PTH=true

LOG_DIR="${SCRIPT_DIR}/logs_v4"
REPORTS_DIR="${SCRIPT_DIR}/reports"
MODELS_DIR="${SCRIPT_DIR}/models_pth"

# ─── Help / Usage ─────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: ./run_all.sh [OPTIONS]

Options:
  --stage <name>         Pipeline stage to run: finetune, all, giaa, pretrain, agg (default: finetune)
  --start-fold <N>       Start execution from fold N (1-5, default: 1)
  --folds <"1 2 3">      Specify custom list of folds (default: "1 2 3 4 5")
  --models <"ICI MIR">   Specify model architectures (default: "ICI MIR")
  --genres <"art ...">   Specify genres (default: "art fashion scenery")
  --batch-size <N>       Batch size for fine-tuning (default: 16)
  --root-dir <path>      Dataset root directory (default: "Dataset")
  --remote <name>        rclone remote name (default: "Google Drive")
  --folder-id <id>       Google Drive folder ID (default: "1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L")
  --force                Force re-run even if outputs already exist
  --dry-run              Display planned execution plan without running
  --no-upload            Disable automated Google Drive synchronization
  --no-keep-pth          Delete local *_finetune.pth after upload (default: keep them locally)
  -h, --help             Show this help message and exit

Examples:
  ./run_all.sh                        # Run PIAA fine-tuning & evaluation across all 5 folds
  ./run_all.sh --stage all            # Run entire 4-stage pipeline (GIAA -> Pretrain -> Finetune -> Agg)
  ./run_all.sh --start-fold 2         # Start pipeline from fold 2 onwards
  ./run_all.sh --models "ICI"         # Run only ICI architecture
  ./run_all.sh --dry-run              # View planned execution queue
EOF
  exit 0
}

# ─── Parse CLI Options ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
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
    --no-keep-pth)
      KEEP_PTH=false
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

# ─── Local Checkpoint Retention ───────────────────────────────────────────────
# KEEP_PTH=true  -> train_PIAA keeps *_finetune.pth, and rclone "copy" leaves the
#                   local files in place after upload (default).
# KEEP_PTH=false -> train_PIAA deletes them after inference, and rclone "move"
#                   removes whatever is left once it reaches Google Drive.
if [ "${KEEP_PTH}" = true ]; then
  PTH_SYNC_ACTION="copy"
else
  PTH_SYNC_ACTION="move"
fi

# ─── Directories Setup ────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}" "${REPORTS_DIR}" "${MODELS_DIR}"

# ─── Signal Traps ─────────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "⚠️ [Pipeline Interrupted] Stopping child processes and exiting cleanly..."
  exit 130
}
trap cleanup SIGINT SIGTERM

# ─── Auto Dataset & Pretrained Models Setup Check ─────────────────────────────
check_dataset() {
  if [ "${DRY_RUN}" = true ]; then
    return 0
  fi
  local test_img="${ROOT_DIR}/sample/art_extracted/art"
  local test_split="${ROOT_DIR}/split/v4_fold1/art/train_PIAA.txt"
  local test_weights="models_pth/v4_fold1/art"

  if [ ! -d "${test_img}" ] || [ ! -f "${test_split}" ] || [ ! -d "${test_weights}" ]; then
    echo "⚠️ [Prerequisite Check] Datasets or pretrained model weights missing in ${ROOT_DIR}/ or models_pth/."
    echo "🚀 Automatically invoking scripts/setup_data.sh to download and prepare files..."
    bash scripts/setup_data.sh
  else
    echo "✅ [Prerequisite Check] Datasets and pretrained model weights verified."
  fi
}

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

# ─── Build Execution Queue ────────────────────────────────────────────────────
ACTIVE_FOLDS=()
for F in "${ALL_FOLDS[@]}"; do
  if [ "${F}" -ge "${START_FOLD}" ]; then
    ACTIVE_FOLDS+=("${F}")
  fi
done

echo "======================================================================"
echo "🚀 XPASS-Simple Automation Pipeline (Stage: ${STAGE})"
echo "======================================================================"
echo "• Active Folds:        ${ACTIVE_FOLDS[*]} (start_fold=${START_FOLD})"
echo "• Architectures:       ${ALL_MODELS[*]}"
echo "• Genres:              ${ALL_GENRES[*]}"
echo "• Batch Size:          ${BATCH_SIZE}"
echo "• Google Drive Remote: ${REMOTE}"
echo "• Target Folder ID:    ${FOLDER_ID}"
echo "• GDrive Sync:         $([ "${NO_UPLOAD}" = true ] && echo "Disabled" || echo "Enabled")"
echo "• Smart Resume:        $([ "${FORCE}" = true ] && echo "Disabled (Force Re-run)" || echo "Enabled")"
echo "======================================================================"
echo ""

check_dataset

START_TIMESTAMP=$(date +%s)
SUCCESS_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
FAILED_JOBS=()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: GIAA Training (NIMA Backbone)
# ══════════════════════════════════════════════════════════════════════════════
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "giaa" ]; then
  echo ""
  echo "======================================================================"
  echo "  [STAGE 1/4] GIAA Training (NIMA Backbone Initialization)"
  echo "======================================================================"
  for F in "${ACTIVE_FOLDS[@]}"; do
    for G in "${ALL_GENRES[@]}"; do
      DATASET_VER="${DATASET_PREFIX}_fold${F}"
      LOG_FILE="${LOG_DIR}/giaa_${G}_fold${F}.log"
      SAMPLES_ROOT="${ROOT_DIR}/sample/${G}_extracted"
      MODEL_DIR="${MODELS_DIR}/${DATASET_VER}/${G}"

      # Check if NIMA model already exists
      if [ "${FORCE}" = false ] && [ -d "${MODEL_DIR}" ] && ls "${MODEL_DIR}"/*NIMA*.pth 1>/dev/null 2>&1; then
        echo "⏩ [GIAA] Skipping existing NIMA checkpoint: ${DATASET_VER} | ${G}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
      fi

      if [ "${DRY_RUN}" = true ]; then
        echo "🔍 [Dry Run] GIAA: ${DATASET_VER} | ${G}"
        continue
      fi

      echo "▶️ [GIAA] Training: ${DATASET_VER} | ${G}"
      if uv run python -m src.train_GIAA \
        --genre "${G}" \
        --dataset_ver "${DATASET_VER}" \
        --root_dir "${ROOT_DIR}" \
        --samples_root "${SAMPLES_ROOT}" \
        --rclone_remote "${REMOTE}" \
        --gdrive_folder_id "${FOLDER_ID}" \
        $([ "${NO_UPLOAD}" = true ] && echo "--no_gdrive_upload") \
        2>&1 | tee "${LOG_FILE}"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        sync_gdrive "${LOG_FILE}" "logs_v4/giaa_${G}_fold${F}.log" "copy"
        sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
      else
        echo "❌ [GIAA Error] Failed on ${DATASET_VER} | ${G}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_JOBS+=("GIAA_${DATASET_VER}_${G}")
      fi
    done
  done
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: PIAA Pretraining (ICI & MIR Architecture Pretrain)
# ══════════════════════════════════════════════════════════════════════════════
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "pretrain" ]; then
  echo ""
  echo "======================================================================"
  echo "  [STAGE 2/4] PIAA Pretraining (ICI / MIR Pretraining on GIAA data)"
  echo "======================================================================"
  for F in "${ACTIVE_FOLDS[@]}"; do
    for M in "${ALL_MODELS[@]}"; do
      for G in "${ALL_GENRES[@]}"; do
        DATASET_VER="${DATASET_PREFIX}_fold${F}"
        LOG_FILE="${LOG_DIR}/${M}_pretrain_${G}_fold${F}.log"
        SAMPLES_ROOT="${ROOT_DIR}/sample/${G}_extracted"
        MODEL_DIR="${MODELS_DIR}/${DATASET_VER}/${G}"

        # Check if pretrain model already exists
        if [ "${FORCE}" = false ] && [ -d "${MODEL_DIR}" ] && ls "${MODEL_DIR}"/*"${M}"*pretrain.pth 1>/dev/null 2>&1; then
          echo "⏩ [Pretrain] Skipping existing pretrain checkpoint: ${DATASET_VER} | ${M} | ${G}"
          SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
          continue
        fi

        if [ "${DRY_RUN}" = true ]; then
          echo "🔍 [Dry Run] Pretrain: ${DATASET_VER} | ${M} | ${G}"
          continue
        fi

        echo "▶️ [Pretrain] Training: ${DATASET_VER} | ${M} | ${G}"
        if uv run python -m src.train_PIAA \
          --genre "${G}" \
          --dataset_ver "${DATASET_VER}" \
          --model_type "${M}" \
          --piaa_mode PIAA_pretrain \
          --batch_size 128 \
          --root_dir "${ROOT_DIR}" \
          --samples_root "${SAMPLES_ROOT}" \
          --rclone_remote "${REMOTE}" \
          --gdrive_folder_id "${FOLDER_ID}" \
          $([ "${NO_UPLOAD}" = true ] && echo "--no_gdrive_upload") \
          2>&1 | tee "${LOG_FILE}"; then
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
          sync_gdrive "${LOG_FILE}" "logs_v4/${M}_pretrain_${G}_fold${F}.log" "copy"
          sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
        else
          echo "❌ [Pretrain Error] Failed on ${DATASET_VER} | ${M} | ${G}"
          FAILED_COUNT=$((FAILED_COUNT + 1))
          FAILED_JOBS+=("Pretrain_${DATASET_VER}_${M}_${G}")
        fi
      done
    done
  done
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: PIAA Fine-Tuning (User-Level Personalization)
# ══════════════════════════════════════════════════════════════════════════════
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "finetune" ]; then
  echo ""
  echo "======================================================================"
  echo "  [STAGE 3/4] PIAA Fine-Tuning (Personalized User Finetune & Evaluation)"
  echo "======================================================================"
  for F in "${ACTIVE_FOLDS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "  📁 Processing Fold: v4_fold${F}"
    echo "----------------------------------------------------------------------"

    for M in "${ALL_MODELS[@]}"; do
      for G in "${ALL_GENRES[@]}"; do
        DATASET_VER="${DATASET_PREFIX}_fold${F}"
        LOG_FILENAME="${M}_finetune_${G}_fold${F}.log"
        LOG_FILE="${LOG_DIR}/${LOG_FILENAME}"
        SAMPLES_ROOT="${ROOT_DIR}/sample/${G}_extracted"

        # Smart Resume Check: check if already completed
        if [ "${FORCE}" = false ] && [ -f "${LOG_FILE}" ]; then
          if grep -q "Evaluation Results" "${LOG_FILE}" 2>/dev/null || grep -q "Test Average" "${LOG_FILE}" 2>/dev/null; then
            echo "⏩ [Finetune] Skipping already completed: ${DATASET_VER} | ${M} | ${G}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
          fi
        fi

        if [ "${DRY_RUN}" = true ]; then
          echo "🔍 [Dry Run] Finetune: ${DATASET_VER} | ${M} | ${G}"
          continue
        fi

        echo "▶️ [Finetune] Running: ${DATASET_VER} | ${M} | ${G}"
        JOB_START_TIME=$(date +%s)

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
          $([ "${KEEP_PTH}" = true ] && echo "--keep_finetune_pth") \
          2>&1 | tee "${LOG_FILE}"; then

          JOB_END_TIME=$(date +%s)
          JOB_DURATION=$((JOB_END_TIME - JOB_START_TIME))
          echo "✅ [Finetune] Completed ${DATASET_VER} | ${M} | ${G} in ${JOB_DURATION}s."
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

          sync_gdrive "${LOG_FILE}" "logs_v4/${LOG_FILENAME}" "copy"
          sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
        else
          echo "❌ [Finetune Error] Failed on ${DATASET_VER} | ${M} | ${G}"
          FAILED_COUNT=$((FAILED_COUNT + 1))
          FAILED_JOBS+=("Finetune_${DATASET_VER}_${M}_${G}")
        fi
      done
    done

    # End-of-Fold Synchronization & Cleanup
    if [ "${DRY_RUN}" = false ]; then
      echo ""
      echo "📦 [Fold v4_fold${F} Completed] Synchronizing reports and logs to Google Drive..."
      sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
      sync_gdrive "${LOG_DIR}/" "logs_v4/" "copy"
      sync_gdrive "${MODELS_DIR}/" "models_pth/" "${PTH_SYNC_ACTION}" "--include *_finetune.pth"
      if [ "${KEEP_PTH}" = true ]; then
        echo "💾 [Retention] Local *_finetune.pth kept in ${MODELS_DIR} (--no-keep-pth to discard)"
      fi
    fi
  done
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Aggregate Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "finetune" ] || [ "${STAGE}" = "agg" ]; then
  echo ""
  echo "======================================================================"
  echo "  [STAGE 4/4] Aggregate Metrics Across All Folds"
  echo "======================================================================"
  for M in "${ALL_MODELS[@]}"; do
    for G in "${ALL_GENRES[@]}"; do
      LOG_FILE="${LOG_DIR}/agg_${M}_${G}.log"
      if [ "${DRY_RUN}" = true ]; then
        echo "🔍 [Dry Run] Aggregate: ${M} | ${G}"
        continue
      fi

      echo "▶️ [Aggregate] Summarizing: ${M} | ${G}"
      uv run python -m src.analysis aggregate \
        --version v4 \
        --genre "${G}" \
        --pattern finetune \
        --method "${M}" \
        2>&1 | tee "${LOG_FILE}" || true

      sync_gdrive "${LOG_FILE}" "logs_v4/agg_${M}_${G}.log" "copy"
    done
  done

  if [ "${DRY_RUN}" = false ]; then
    sync_gdrive "${REPORTS_DIR}/" "reports/" "copy"
  fi
fi

# ─── Execution Summary ────────────────────────────────────────────────────────
END_TIMESTAMP=$(date +%s)
TOTAL_DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "======================================================================"
echo "🎉 PIPELINE SUMMARY (Stage: ${STAGE})"
echo "======================================================================"
echo "• Total Elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "• Succeeded:     ${SUCCESS_COUNT}"
echo "• Skipped:       ${SKIPPED_COUNT}"
echo "• Failed:        ${FAILED_COUNT}"

if [ "${FAILED_COUNT}" -gt 0 ]; then
  echo "⚠️ Failed Jobs:   ${FAILED_JOBS[*]}"
  exit 1
else
  echo "✅ All pipeline tasks finished successfully!"
  exit 0
fi
