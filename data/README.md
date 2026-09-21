# Dataset preparation

## 1. Video-MME Long

Annotations come from
[topyun/Video-MME-Long](https://huggingface.co/datasets/topyun/Video-MME-Long).
Videos come from the
[official Video-MME dataset](https://huggingface.co/datasets/lmms-lab/Video-MME).

Install `huggingface_hub` and `pyarrow`, then run from the repository root:

```bash
pip install --upgrade huggingface_hub pyarrow
bash scripts/prepare_videomme_long.sh
```

The script downloads the data, converts the annotation Parquet to JSON,
extracts the 300 long videos, and validates the result.

```text
data/raw/videomme/
├── videoMME_long.json
└── videos/                 # 300 videos
```

## 2. M3-Bench

Download the robot videos from the [ByteDance-Seed/M3-Bench dataset](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench).
Download[`robot.json`](https://github.com/ByteDance-Seed/m3-agent/tree/master/data/annotations)from the from m3-agent repository. 

Place the MP4 files from `videos/robot/` and the annotation as
follows:

```text
data/raw/m3bench/
├── annotations/robot.json
└── videos/
```

Preprocessing outputs are written under `data/processed/`.
