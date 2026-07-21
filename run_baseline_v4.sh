#!/usr/bin/env bash
# รัน ICI/MIR บน v4 (leak-free) ครบ pipeline — รันในโฟลเดอร์ XPASS-Simple/
# ต้องมี Dataset/split/v4_fold{1..5}/ อยู่ก่อน (ก๊อปจาก split_v4_xpass)
# resume ได้: XPASS เขียน checkpoint .pth ต่อ fold; รันซ้ำจะข้ามที่เสร็จแล้ว
set -e
GENRES=(art fashion scenery)
ROOT=Dataset
# samples_root ต่อ genre (layout จริง: Dataset/sample/{genre}_extracted/...)
declare -A SR=(
  [art]=Dataset/sample/art_extracted
  [fashion]=Dataset/sample/fashion_extracted
  [scenery]=Dataset/sample/scenery_extracted
)
LOG=logs_v4; mkdir -p $LOG

echo "===== [1/3] GIAA (prerequisite) ====="
for G in "${GENRES[@]}"; do
  echo ">> GIAA $G"
  python -m src.train_GIAA --genre "$G" --dataset_ver v4_all \
    --root_dir "$ROOT" --samples_root "${SR[$G]}" 2>&1 | tee "$LOG/giaa_$G.log"
done

for M in ICI MIR; do
  echo "===== [2/3] PIAA $M ====="
  for G in "${GENRES[@]}"; do
    echo ">> $M pretrain $G"
    python -m src.train_PIAA --genre "$G" --dataset_ver v4_all --model_type "$M" \
      --piaa_mode PIAA_pretrain --batch_size 128 \
      --root_dir "$ROOT" --samples_root "${SR[$G]}" 2>&1 | tee "$LOG/${M}_pretrain_$G.log"
    echo ">> $M finetune $G"
    python -m src.train_PIAA --genre "$G" --dataset_ver v4_all --model_type "$M" \
      --piaa_mode PIAA_finetune --batch_size 16 \
      --root_dir "$ROOT" --samples_root "${SR[$G]}" 2>&1 | tee "$LOG/${M}_finetune_$G.log"
  done
done

echo "===== [3/3] Aggregate (CCC/SROCC ต่อ genre) ====="
for M in ICI MIR; do
  for G in "${GENRES[@]}"; do
    python -m src.analysis aggregate --version v4 --genre "$G" --pattern finetune --method "$M" \
      2>&1 | tee "$LOG/agg_${M}_$G.log"
  done
done
echo "DONE — CCC เฉลี่ยอยู่ใน $LOG/agg_*.log เอาไปใส่ Table 1 (แทนเลข v3)"
