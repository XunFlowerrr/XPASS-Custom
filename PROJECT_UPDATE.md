# XPASS-Custom — Development and Experiment Update

เอกสารนี้สรุปงานที่ทำใน `XunFlowerrr/XPASS-Custom` นับตั้งแต่แยกออกจาก
`pinwap/XPASS-Simple` จนถึงการรัน PIAA finetuning ครบทุกชุดทดลอง

## Repository Status

- Upstream: `pinwap/XPASS-Simple` (`origin`)
- Fork: `XunFlowerrr/XPASS-Custom` (`xun`)
- Development branch: `perf/finetune-checkpoint-slimming`
- จุดที่เริ่มแยกจาก upstream: `3fc0d1a`
- มีการพัฒนาเพิ่มจาก `origin/main` 18 commits ณ เวลาที่จัดทำเอกสาร
- การเปลี่ยนแปลงรวมประมาณ 6,514 บรรทัดเพิ่ม และ 147 บรรทัดลบ ใน 33 ไฟล์

## 1. Environment and Device Support

ปรับโครงการให้จัดการ environment ด้วย `uv` โดยเพิ่ม `pyproject.toml`,
`uv.lock` และ `.python-version` พร้อมเพิ่มการเลือก device ผ่าน `--device`

- รองรับ CUDA, CPU และ Apple MPS
- ปรับ `autocast`, `GradScaler` และ `torch.load(map_location=...)` ให้สัมพันธ์กับ device
- เพิ่ม automated tests สำหรับ device compatibility
- แก้ `datetime` import ที่หายไปใน `train_PIAA.py` ซึ่งทำให้ PIAA หยุดก่อนเริ่มเทรน

## 2. Progress Tracking and ETA

เพิ่มระบบติดตามความคืบหน้าหลายระดับ ได้แก่ genre, fold, model/user, epoch,
train/validation phase, elapsed time และ estimated remaining time

ตัวอย่าง output:

```text
[art 1/1 | v4_fold3 1/1 | user 131 21/23 |
 Ep 104/200 [Train] | Elapsed: ... | ETA: ...]
```

มี automated tests สำหรับ progress และ ETA เพื่อป้องกัน regression

## 3. End-to-End Pipeline

เพิ่ม `run_all.sh` เพื่อควบคุม pipeline สี่ขั้นตอน:

1. GIAA training
2. PIAA pretraining
3. PIAA per-user finetuning
4. Metric aggregation

ความสามารถที่เพิ่มเข้ามา:

- เลือก fold, genre และ model ได้
- รองรับ ICI และ MIR
- รองรับ dry run, force rerun และระบุ fold เริ่มต้น
- กำหนด batch size, worker และ pixel-cache budget ได้
- เก็บ log แยกตาม job
- sync report, log และ checkpoint ขึ้น Google Drive
- aggregate metrics หลัง finetune
- รองรับการกำหนดลำดับ fold เช่น `--folds "5 4 3"`

## 4. Dataset and Pretrained Model Setup

เพิ่ม `scripts/setup_data.sh` สำหรับ:

- ติดตั้ง Python environment
- ดาวน์โหลด XPASS dataset essentials และรูปภาพ
- ดาวน์โหลด pretrained weights ของ art, fashion และ scenery
- แตกไฟล์และจัด directory layout
- ตรวจสอบ split, dataset และ model ก่อนเริ่มรัน

ปรับความปลอดภัยของขั้นตอน download เพิ่มเติม:

- ไม่ถือว่าไฟล์ที่เพียงแค่ไม่ว่างเป็น archive ที่สมบูรณ์
- ตรวจขั้นต่ำของขนาดไฟล์และตรวจ archive ด้วย `unzip -t`
- ตรวจจับกรณี Google Drive ส่ง HTML quota/rate-limit page แทนไฟล์จริง
- เพิ่ม `--skip-models`
- ลบ archive ที่แตกแล้วเพื่อลด storage ประมาณ 16.7 GB
- ลบ archive หลัง verification เท่านั้น เพื่อรักษา recovery path

## 5. Finetune Checkpoint Slimming

เดิม checkpoint ต่อ user เก็บ CLIP/NIMA backbone ที่ถูก freeze ซ้ำในทุกไฟล์

| | ขนาดโดยประมาณต่อ user |
|---|---:|
| Full checkpoint เดิม | 346 MB |
| Trainable delta checkpoint ใหม่ | 7 MB |

ระบบใหม่เก็บเฉพาะ trainable state แล้วประกอบกับ frozen NIMA weights จาก
pretrain checkpoint ตอนโหลด พร้อมรองรับ full checkpoint จากเวอร์ชันเก่า

ผลโดยประมาณสำหรับ 780 user checkpoints:

- เดิมประมาณ 270 GB
- ใหม่ประมาณ 5.6 GB
- ลดขนาดประมาณ 48 เท่า

มี tests ยืนยันว่า `pretrain + delta` ให้ model state เท่ากับ full checkpoint
ทุก tensor ด้วย `torch.equal` จึงไม่เปลี่ยนผลทางคณิตศาสตร์ของโมเดล

## 6. Model and Pretrain-State Reuse

เดิมระบบสร้าง CLIP ViT-B/16 และอ่าน pretrain checkpoint ใหม่สำหรับทุก user
จึงปรับเป็น:

- สร้าง PIAA model ครั้งเดียวต่อ job
- โหลด pretrain state ครั้งเดียว
- reset model กลับสู่ base state ก่อนเริ่ม user ถัดไป
- fallback ไปสร้างโมเดลใหม่หาก checkpoint ไม่ครอบคลุม frozen keys ครบ
- ลบ `.item()` ที่ไม่ถูกใช้งานออกจาก PIAA forward เพื่อลด CUDA synchronization

## 7. Data Pipeline and GPU Utilization

การวัดบน NVIDIA L4 พบว่า CPU data pipeline เป็นคอขวด:

- Decode + transform: ประมาณ 139 images/s/core
- Frozen ViT-B/16 บน L4: ประมาณ 825 images/s
- `num_workers=4` เดิมป้อนข้อมูลให้ GPU ไม่ทัน

การปรับปรุงประกอบด้วย:

- เพิ่ม `make_loader()` เพื่อรวม DataLoader configuration
- เปิด `persistent_workers` สำหรับ loader ที่ใช้ข้าม epoch
- เปิด `pin_memory` บน CUDA และเพิ่ม `prefetch_factor`
- ตั้ง default workers เป็น `CPU count - 2` สูงสุด 8
- จำกัด OMP/MKL threads เมื่อรันหลาย shard
- เพิ่ม decoded-pixel cache ใน RAM
- JPEG ถูก decode ครั้งเดียว แต่ augmentation ยังทำงานใหม่ทุก access

ผล benchmark สำหรับหนึ่ง user จำลอง:

| Configuration | เวลา |
|---|---:|
| เดิม: 4 workers | 7.01 s |
| Persistent workers + pin/prefetch | 4.47 s |
| เพิ่ม decoded-pixel cache | 3.40 s |

Training loop เร็วขึ้นประมาณ 2.06 เท่า โดยมี tests ยืนยันว่า cached และ
uncached pixels รวมถึง transform output ภายใต้ seed เดียวกันยังเท่ากัน

## 8. Concurrent Jobs per GPU

เพิ่ม flags สำหรับแบ่งงานและรันพร้อมกัน:

- `--jobs N`
- `--shard i/N`
- `--num-workers N`
- `--pixel-cache-gb F`
- `--no-agg`
- `--skip-setup`

ผลวัดจริงบน L4:

| รูปแบบ | Throughput รวม |
|---|---:|
| 1 job, 6 workers | 305 images/s |
| 2 jobs, 3 workers/job | 465 images/s |
| 2 jobs, 4 workers/job | **546 images/s** |
| 3 jobs, 2 workers/job | 541 images/s |

สรุปว่า 2 jobs × 4 workers เป็น configuration ที่ดีที่สุดบนเครื่อง
8-vCPU + L4 โดย throughput เพิ่มประมาณ 1.79 เท่า

Shard ถูกแบ่งแบบ round-robin และมี tests ยืนยันว่า:

- ไม่มี job ซ้ำระหว่าง shard
- งานรวมครบทุก combination
- shard assignment ไม่เปลี่ยนหลัง resume

## 9. Smart Resume

Smart resume เดิมตรวจ log ด้วยข้อความ `Evaluation Results` หรือ `Test Average`
แต่ไม่มีข้อความทั้งสองอยู่ใน source code จึงไม่มี job ใดถูกข้ามจริง

ระบบใหม่ตรวจ:

- Report JSON ที่ job สร้างจริง
- หรือข้อความ `Test results saved to` ใน log

ทำให้สามารถหยุดและรันต่อ ข้ามงานที่เสร็จแล้ว และใช้ reports ที่ sync จาก
Google Drive เป็น resume state ระหว่างเครื่องได้

## 10. Checkpoint Upload and Retention

เพิ่ม Google Drive checkpoint upload callback ที่รองรับ:

- atomic checkpoint save และ integrity verification
- retry/fail-safe
- background upload queue แบบ non-blocking
- รอ pending uploads ก่อน process จบ
- เลือกเก็บหรือลบ local checkpoint ได้

ค่า default ปัจจุบันเก็บ `*_finetune.pth` ในเครื่องและใช้ `rclone copy`
แทน `move` เพื่อป้องกัน local checkpoint หายโดยไม่ตั้งใจ

End-of-fold sync ถูกจำกัดตาม `(fold, model, genre)` เพื่อป้องกัน shard หนึ่ง
ย้าย checkpoint ของอีก shard ระหว่างที่รันพร้อมกัน

## 11. Lightning Studio Support

เพิ่ม:

- `scripts/bootstrap_studio.sh`
- `scripts/run_fold.sh`
- `fleet/` package และ CLI สำหรับ Lightning SDK

ข้อค้นพบจาก Lightning Studio:

- Studio duplicate สามารถ copy filesystem state ได้
- `$HOME` รอดจาก stop/start
- `rclone` ที่ติดตั้งใน `/usr/bin` ไม่รอดจาก restart
- `.venv` ที่ชี้ไป `/system/conda/...` ไม่ควรถูกถือว่า persistent
- bootstrap จึงติดตั้ง `rclone` ใน `$HOME/bin`, ติดตั้ง `uv` ใน
  `$HOME/.local/bin` และสร้าง environment ใหม่เมื่อจำเป็น

มีการสร้าง fleet orchestrator สำหรับ duplicate/start/stop/dispatch/monitor
แต่ **ไม่ได้ใช้ orchestrator เป็นตัวรัน experiment รอบนี้** เนื่องจาก workflow
ยังไม่พร้อมสำหรับการใช้งานจริง การรันรอบนี้ใช้วิธี manual ผ่าน SSH, tmux,
`run_all.sh`, reverse fold order และสอง shards ต่อ L4

## 12. Automated Tests

เพิ่ม tests ครอบคลุม:

- Device compatibility
- Progress และ ETA
- Checkpoint callback และ background upload
- Delta checkpoint และ model-reuse equivalence
- Pixel cache equivalence
- DataLoader worker behavior
- Shard partition และ resume stability
- Fleet state persistence

ผลล่าสุดระหว่างพัฒนา:

```text
56 passed
```

## 13. Experiment Execution

รัน PIAA finetuning สำหรับ:

- 5 folds
- 3 genres: art, fashion และ scenery
- 2 models: ICI และ MIR

รวมทั้งหมด:

```text
5 folds × 3 genres × 2 models = 30 jobs
```

กลยุทธ์การรันจริง:

- เครื่องแรกวิ่ง fold จากหน้าไปหลัง
- เครื่องที่สองวิ่ง fold จากหลังมาหน้า
- เครื่องที่สองรันสอง shards พร้อมกันผ่าน tmux
- ใช้ Google Drive เป็นศูนย์รวม report และ checkpoint
- ใช้ report JSON เป็น resume marker

## 14. Using the New Finetune Inference Path

Finetune checkpoint รูปแบบใหม่เป็น **trainable delta checkpoint** ไม่ใช่
full model checkpoint ดังนั้นการนำไป inference ต้องมีไฟล์สองส่วนที่ตรงกัน:

1. Shared pretrain checkpoint: `*_{ICI|MIR}_*_pretrain.pth`
2. Per-user delta checkpoint: `*_{ICI|MIR}_user_<uid>_*_finetune.pth`

ทั้งสองไฟล์ต้องมาจาก fold, genre และ model type เดียวกัน เช่น delta ของ
`v4_fold3/art/ICI/user_131` ต้องประกอบกับ ICI pretrain ของ
`v4_fold3/art` เท่านั้น ห้ามใช้ pretrain ข้าม fold, genre หรือข้าม ICI/MIR

### 14.1 Training Followed by Automatic Inference

`src.train_PIAA` เรียก test inference ให้อัตโนมัติหลัง finetuning เสร็จ
ไม่ต้องสั่ง inference เพิ่ม ตัวอย่างสำหรับ ICI:

```bash
uv run python -m src.train_PIAA \
  --genre art \
  --dataset_ver v4_fold3 \
  --model_type ICI \
  --piaa_mode PIAA_finetune \
  --batch_size 16 \
  --root_dir Dataset \
  --samples_root Dataset/sample/art_extracted \
  --keep_finetune_pth
```

สำหรับ MIR เปลี่ยนเป็น `--model_type MIR`

ระบบจะทำตามลำดับต่อไปนี้:

1. ค้นหา `*_pretrain.pth` ที่ตรงกับ fold, genre และ model type
2. Finetune โมเดลแยกตาม user และบันทึกเฉพาะ trainable delta
3. สร้าง PIAA model สำหรับ inference
4. โหลด frozen/shared state จาก pretrain checkpoint
5. โหลด per-user delta ทับด้วย `strict=False`
6. ประเมิน test split และเขียน report JSON ลง
   `reports/exp/<dataset_ver>/<genre>/`

`--keep_finetune_pth` ใช้เมื่อจำเป็นต้องเก็บ per-user delta สำหรับ inference
ในอนาคต หากไม่ใส่ flag นี้ `train_PIAA` จะลบ local finetune checkpoints
หลัง inference เสร็จ แม้ report JSON จะยังคงอยู่

### 14.2 Loading One User Without Finetuning Again

CLI `src.train_PIAA --piaa_mode PIAA_finetune` ยังเป็น train-then-infer workflow
ไม่ใช่ inference-only command หากต้องการโหลด checkpoint ที่มีอยู่แล้วเพื่อ
predict โดยไม่เทรนซ้ำ ให้ประกอบ base state กับ user delta ดังนี้:

```python
from argparse import Namespace

import torch

from src.train_common import (
    build_piaa_model,
    is_trainable_only_state,
    num_bins,
)

genre = "art"
model_type = "ICI"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# XPASS v4 dimensions; derive these from the dataset for another schema.
num_attr = 45   # QIP dimensions
num_pt = 116    # personal-trait dimensions

args = Namespace(model_type=model_type, dropout=0.1)
model = build_piaa_model(
    num_bins=num_bins,
    num_attr=num_attr,
    num_pt=num_pt,
    genres=[genre],
    backbone_dict={genre: "clip_vit_b16"},
    args=args,
).to(device)

pretrain_path = (
    "models_pth/v4_fold3/art/"
    "art_ICI_Only_<timestamp>_pretrain.pth"
)
user_delta_path = (
    "models_pth/v4_fold3/art/"
    "art_ICI_user_131_Only_<timestamp>_finetune.pth"
)

base_state = torch.load(pretrain_path, map_location=device)
user_state = torch.load(user_delta_path, map_location=device)

if is_trainable_only_state(user_state):
    # New format: shared frozen state + per-user trainable delta.
    model.load_state_dict(base_state, strict=False)
    model.load_state_dict(user_state, strict=False)
else:
    # Backward compatibility for old full finetune checkpoints.
    model.load_state_dict(user_state)

model.eval()

# Inputs must use the same CLIP transform and XPASS encoders as training:
# image  [B, 3, 224, 224]
# traits [B, 116]
# qip    [B, 45]
with torch.no_grad():
    prediction = model(
        image.to(device),
        traits.float().to(device),
        qip.float().to(device),
        genre,
    )
```

หาก `user_state` เป็น checkpoint เก่าแบบ full model ระบบจะโหลดไฟล์นั้นตรง ๆ
เพื่อรักษา backward compatibility ส่วน checkpoint ใหม่ต้องมี pretrain file
อยู่ด้วยเสมอ การย้ายหรือสำรองโมเดลจึงควรเก็บ pretrain checkpoint และ user
delta checkpoints ของแต่ละ `(fold, genre, model type)` ไว้ด้วยกัน

### 14.3 Re-running Test Inference for All Users

ฟังก์ชัน `src.inference.inference_finetune(...)` รองรับ checkpoint ใหม่แล้ว
โดยจะโหลด pretrain state ครั้งเดียวและนำ user delta แต่ละไฟล์มาทับก่อนประเมิน
user นั้น นอกจากนี้ยังตรวจ checkpoint เก่าและโหลดแบบ full state ให้อัตโนมัติ

ข้อควรระวังคือชื่อไฟล์ user checkpoint มี `experiment_name`/timestamp อยู่ในชื่อ
ดังนั้น caller ต้องส่ง `experiment_name` ที่ตรงกับชุด checkpoint ที่ต้องการ
อ่าน มิฉะนั้นระบบจะรายงานว่าไม่พบ best model ของ user นั้น

## Final Status

ตรวจ Google Drive แล้วพบ report ครบทุกชุด:

```text
v4_fold1: 6/6
v4_fold2: 6/6
v4_fold3: 6/6
v4_fold4: 6/6
v4_fold5: 6/6
```

รวมทั้งหมด:

```text
30/30 finetune report JSON files
```

- ครบทุก combination ของ fold × genre × model
- ไม่มี experiment combination ตกหล่น
- Reports พร้อมสำหรับ aggregate SROCC, PLCC, MAE, NDCG@10 และ CCC
- Delta checkpoints ถูกจัดเก็บและ sync ตาม configuration ที่กำหนด

## Summary

หลังแยกจาก upstream งานหลักที่ดำเนินการแล้วคือ:

1. ทำให้ repository reproducible ด้วย `uv`
2. เพิ่ม progress, ETA และ monitoring
3. สร้าง setup และ end-to-end pipeline
4. แก้ resume ให้ใช้งานได้จริง
5. ลด checkpoint จากประมาณ 346 MB เหลือประมาณ 7 MB/user
6. ลดการสร้างและโหลด CLIP ซ้ำ
7. แก้ CPU/DataLoader bottleneck
8. รองรับสองงานพร้อมกันต่อ L4
9. เพิ่ม Google Drive upload ที่ปลอดภัยและไม่ block training
10. เพิ่ม Lightning Studio bootstrap และ orchestration groundwork
11. รัน experiment ครบ 30/30 jobs และตรวจความครบถ้วนบน Google Drive
