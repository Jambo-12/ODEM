# Preprocessing

ODEM publishes preprocessing support for **VideoMME-Long** and
**M3-Bench-robot**. Both benchmarks use the same pipeline:

1. generate temporal captions with Qwen2.5-VL-7B-Instruct;
2. encode the four caption fields with `text-embedding-3-large`;
3. encode the same temporal segments with ViCLIP.

## 1. Repository layout

Place models and datasets directly in the repository.

```text
models/
├── Qwen2.5-VL-7B-Instruct/
└── ViCLIP/ViClip-InternVid-10M-FLT.pth

data/raw/
├── videomme/videos/
└── m3bench/videos/
```

## 2. API configuration

```bash
cp .env.example .env
```

Fill in the required API keys in the repository-level `.env` file.

## 3. Run preprocessing

```bash
bash scripts/preprocess/videomme.sh
bash scripts/preprocess/m3bench.sh
```

Both pipelines use one GPU. Override the default GPU when needed:

```bash
PREPROCESS_GPU_ID=1 bash scripts/preprocess/videomme.sh
bash scripts/preprocess/m3bench.sh --gpu_id 1 --video_name living_room_06
```

## 4. Output layout

```text
data/processed/<benchmark>/<preprocess_id>/
├── manifest.json
└── <video_id>/
    ├── captions.json
    ├── segment_textual_embedding.pkl
    └── segment_visual_embedding.pkl
```

The three per-video files preserve the original ODEM schemas and processing
semantics. `manifest.json` only records resolved paths, models, parameters, and
completion counts for inspection and reproducibility.
