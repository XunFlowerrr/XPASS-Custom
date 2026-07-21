<p align="center">
  <img src="logo.png" alt="proj-xpass-logo" width="750">
</p>

## 概要

**XPASS-SIMPLE** は、個人化画像美的評価（PIAA）研究のためのベースコードです。XPASS データセット
（アート作品・ファッション画像・風景の3ドメイン）を用い、各ドメイン**内**での一般画像美的評価
（GIAA）モデルと個人化美的評価（PIAA）モデルの学習・推論を行います。

---

## 目次

1. [環境構築](#環境構築)
2. [事前に用意するデータ](#事前に用意するデータ)
3. [学習（GIAA）](#学習giaa)
4. [学習（PIAA）](#学習piaa)
5. [結果の集約（aggregate）](#結果の集約aggregate)
6. [特徴量の次元構成](#特徴量の次元構成)
7. [コミットメッセージ規則](#コミットメッセージ規則)
8. [データ統計](#データ統計)

---

## 環境構築

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```
---

## 事前に用意するデータ

### メタデータ（`data/maked/`）

| ファイル | 内容 |
|---|---|
| `users.csv` | ユーザープロファイル（`user_id` をキーに人口統計・性格・興味） |
| `ratings.csv` | サンプルごとの評価（`user_id`, `sample_file`, `genre`, `Aesthetic`, ...） |
| `QIP_{genre}.csv` | ジャンル別の画像知覚品質特徴（art / fashion）。scenery は `QIP_scenery_image.csv` |

### 画像（`data/samples/`）

- art / fashion: `data/samples/{genre}/{ファイル名}`
- scenery: `data/samples/scenery_image/{ファイル名}`（`.mp4` 名のサンプルは `.jpg` を参照）

### 分割ファイル（`data/split/{dataset_ver}/{genre}/`）

`--dataset_ver`（例 `v3_fold1`）と `--genre` で参照されます。

- `train_images_GIAA.txt` / `val_images_GIAA.txt` — GIAA 学習・検証用画像リスト
- `train_users_GIAA.txt` / `val_users_GIAA.txt` — GIAA pretrain 用ユーザーリスト
- `train_PIAA.txt` / `val_PIAA.txt` / `test_PIAA.txt` — PIAA 用（形式: `user_id\tfilename`）

> **注意:** GIAA プールと PIAA プールは分離されており、`train/val_giaa_dataset` は
> `train/val/test_PIAA.txt` に含まれる user_id を自動的に除外して PIAA ユーザーのリークを防ぎます。

---

## 学習（GIAA）

NIMA（バックボーン + 美的スコアヘッド）を EMD 損失で学習します。学習・検証・テストはすべて指定した単一ジャンル（`--genre`）の中で完結します。

#### 主なオプション引数

| 引数 | 型 | デフォルト | 説明 |
|------|------|------|------|
| `--genre` | str | (必須) | 学習ジャンル（`art` / `fashion` / `scenery` / `all`） |
| `--dataset_ver` | str | `v3_all` | データ分割バージョン（`_all` で終わると全foldを順次実行） |
| `--backbone` | str | `clip_vit_b16` | バックボーン（`clip_vit_b16` のみ） |
| `--root_dir` | str | `{repo}/data` | 画像データのルートディレクトリ |
| `--num_epochs` | int | `200` | 最大エポック数 |
| `--batch_size` | int | `32` | バッチサイズ |
| `--lr` | float | `1e-5` | 学習率 |
| `--lr_decay_factor` | float | `0.5` | ReduceLROnPlateauの減衰率 |
| `--lr_patience` | int | `5` | ReduceLROnPlateauのpatience |
| `--max_patience_epochs` | int | `10` | Early stoppingの忍耐エポック数 |
| `--dropout` | float | `0.1` | ドロップアウト率 |
| `--num_workers` | int | `4` | DataLoaderのワーカー数 |

#### コマンド例

```bash
# art
python -m src.train_GIAA --genre art

# 全ジャンルを順次学習
python -m src.train_GIAA --genre all
```

学習済みモデルは `models_pth/{dataset_ver}/{genre}/` に、テスト結果JSONは `reports/exp/{dataset_ver}/{genre}/` に保存されます。

---

## 学習（PIAA）

GIAA 学習済み NIMA を初期値として、個人化美的評価モデル（`ICI` または `MIR`）を学習します。
`PIAA_pretrain`（全ユーザー共通モデルの事前学習）→ `PIAA_finetune`（ユーザーごとに微調整）の2段階です。
損失関数は MSE で固定です。

#### 主なオプション引数

| 引数 | 型 | デフォルト | 説明 |
|------|------|------|------|
| `--genre` | str | (必須) | 学習ジャンル（`art` / `fashion` / `scenery` / `all`） |
| `--dataset_ver` | str | `v3_all` | データ分割バージョン |
| `--piaa_mode` | str | `PIAA_pretrain` | PIAAモード（`PIAA_pretrain` / `PIAA_finetune`） |
| `--model_type` | str | `ICI` | PIAAモデル（`ICI`: インタラクションベース / `MIR`: MLP Interaction Regression） |
| `--backbone` | str | `clip_vit_b16` | バックボーン（`clip_vit_b16` のみ） |
| `--root_dir` | str | `{repo}/data` | 画像データのルートディレクトリ |
| `--num_epochs` | int | `200` | 最大エポック数 |
| `--batch_size` | int | `32`（pretrain）/ `16`（finetune） | バッチサイズ（未指定時はモードに応じて自動設定） |
| `--lr` | float | `5e-6`（pretrain）/ `1e-5`（finetune） | 学習率（未指定時はモードに応じて自動設定） |
| `--lr_decay_factor` | float | `0.5` | ReduceLROnPlateauの減衰率 |
| `--lr_patience` | int | `5` | ReduceLROnPlateauのpatience |
| `--max_patience_epochs` | int | `10` | Early stoppingの忍耐エポック数 |
| `--dropout` | float | `0.1` | ドロップアウト率 |
| `--num_workers` | int | `4` | DataLoaderのワーカー数 |
| `--start_fold` | int | `1` | 再開するfold番号（`--dataset_ver` が `_all` の場合に使用） |
| `--no_save_model` | flag | `False` | モデルをディスクに保存せず最良モデルをメモリに保持 |
| `--keep_finetune_pth` | flag | `False` | finetune後に `*_finetune.pth` を削除せず残す |

#### コマンド例

```bash
# Pretrain
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --piaa_mode PIAA_pretrain --batch_size 128

# Finetune
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --piaa_mode PIAA_finetune --batch_size 16

# MIR: Pretrain / Finetune
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --model_type MIR --piaa_mode PIAA_pretrain --batch_size 128
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --model_type MIR --piaa_mode PIAA_finetune --batch_size 16
```

> **前提:** PIAA_pretrain は `models_pth/{dataset_ver}/{genre}/` に GIAA 学習済み NIMA（`*NIMA*.pth`）が
> 存在することを前提とします。PIAA_finetune は同ディレクトリの `*_pretrain.pth` を読み込みます。

結果JSONは `reports/exp/{dataset_ver}/{genre}/` に保存されます。

---

## 結果の集約（aggregate）

各 fold の推論結果 JSON を集約し、PIAA 指標（SROCC / NDCG@10 / MAE / CCC）の
fold 平均・全ユーザー平均と標準偏差を出力します。クロスバリデーション結果の評価に使います。

```bash
# v3 の全 fold から ICI finetune 結果を集約
python -m src.analysis aggregate --version v3 --genre art --pattern finetune --method ICI
```

#### 主なオプション引数

| 引数 | 型 | デフォルト | 説明 |
|------|------|------|------|
| `--version` | str | (必須) | データ分割バージョン（例: `v3`） |
| `--genre` | str | (必須) | ジャンル（`art` / `fashion` / `scenery`） |
| `--pattern` | str | `""` | JSONファイル名のグロブパターン（例: `finetune`, `pretrain`） |
| `--method` | str | `None` | モデルでフィルタ（`ICI` / `MIR`） |
| `--folds` | int+ | `None` | 集約する fold 番号（例: `--folds 1 3 5`）。省略で全 fold |
| `--ids` / `--min-id` / `--max-id` | int | `None` | run ID で絞り込み |
| `--reports_dir` | str | `{repo}/reports/exp` | 結果JSONのルートディレクトリ |

---

## 特徴量の次元構成

### 個人特性ベクトル（116次元）

`traits` ベクトルはユーザー固有の特性と嗜好を表現し、2つのカテゴリに分かれた116次元で構成されます。

#### 1. スコアベクトル（70次元）

性格および興味に関するアンケート回答。各質問は7段階（0-6）で評価され、ワンホットエンコーディングにより1問あたり7次元となります。

- **Q1-Q10**（70次元）：ビッグファイブ性格モデルに基づく10問の性格特性質問
- 興味フィールド（art_interest, fashion_interest, photoVideo_interest）も7次元のワンホットベクトルとして合計70次元に含まれます。

#### 2. 属性ベクトル（46次元）

| 属性 | 次元数 | 説明 |
|------|--------|------|
| age_onehot | 5 | 年齢グループ（5区間） |
| gender_onehot | 3 | 性別（3カテゴリ） |
| edu_onehot | 7 | 学歴（7カテゴリ） |
| nationality_onehot | 4 | 国籍（4カテゴリ） |
| art_learn_onehot | 2 | 芸術学習経験（有/無） |
| fashion_learn_onehot | 2 | ファッション学習経験（有/無） |
| photoVideo_learn_onehot | 2 | 写真/映像学習経験（有/無） |

**合計: 70 + 46 = 116次元**

### 画像知覚品質ベクトル - QIP（45次元）

`QIP` ベクトルは各画像から抽出された客観的視覚特徴を含みます。

| カテゴリ | 次元 | 内容 |
|---|---|---|
| 基本画像特性 | 6 | 画像サイズ、アスペクト比、RMSコントラスト、輝度エントロピー、複雑さ、エッジ密度 |
| 色特性 | 20 | 色エントロピー、RGB/Lab/HSV の平均・標準偏差 |
| 構図とバランス | 6 | 鏡像対称性、DCM距離、DCM位置(x,y)、バランス |
| 対称性特徴 | 3 | CNN対称性（左右 / 上下 / 複合） |
| テクスチャと周波数特性 | 8 | フーリエ勾配・シグマ、2D/3Dフラクタル次元、自己相似性（PHOG/CNN）、異方性、均質性 |
| 視覚的複雑さ | 3 | 1次/2次EOE、スパース性、変動性 |

**合計: 45次元**（img_file列を除く）

---

## コミットメッセージ規則

| プレフィックス | 用途 |
|----------------|------|
| `feat:` | 新機能やモジュールの追加 |
| `fix:` | バグや不具合の修正 |
| `refactor:` | 内部構造やコードの再構成（動作変更なし） |
| `exp:` | 実験関連ファイルの追加・更新 |
| `data:` | データファイルの追加・更新 |
| `docs:` | ドキュメントの更新 |
| `conf:` | 設定ファイルの変更 |
| `chore:` | その他の雑務（依存関係の更新、`.gitignore` の修正など） |

---
