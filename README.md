# ODEM: Agentic Long-Video Understanding with Episodic Memory

ODEM builds timestamped episodic memory for long-video question answering.

This release supports Video-MME Long and M3-Bench.

## 1. Installation

Python 3.10, CUDA, `ffmpeg`, and `ffprobe` are required.

```bash
git clone https://github.com/Jambo-12/ODEM.git
cd ODEM

conda create -n odem python=3.10 -y
conda activate odem
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 2. Models

Download
[Qwen2.5-VL-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
and [ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP), then use this layout:

```text
models/
├── Qwen2.5-VL-7B-Instruct/
└── ViCLIP/ViClip-InternVid-10M-FLT.pth
```

## 3. Datasets

Prepare Video-MME Long with:

```bash
pip install --upgrade huggingface_hub pyarrow
bash scripts/prepare_videomme_long.sh
```

For M3-Bench, download the robot videos from
[ByteDance-Seed/M3-Bench](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)
and download
[`robot.json`](https://github.com/ByteDance-Seed/m3-agent/blob/master/data/annotations/robot.json).
Place them under:

```text
data/raw/m3bench/
├── annotations/robot.json
└── videos/
```

See [`data/README.md`](data/README.md) for the complete dataset layout.

## 4. API configuration

```bash
cp .env.example .env
```

Fill in the required API keys in `.env`.

## 5. Preprocessing

```bash
bash scripts/preprocess/videomme.sh
bash scripts/preprocess/m3bench.sh
```

Outputs are written under `data/processed/`. See
[`scripts/preprocess/README.md`](scripts/preprocess/README.md) for optional
arguments and output files.

## 6. Evaluation

```bash
bash scripts/run_videomme.sh
bash scripts/run_m3bench.sh
```

M3-Bench requires a separate semantic evaluation step:

```bash
python -m eval.m3bench.evaluate
```

See [`eval/README.md`](eval/README.md) for the single-GPU evaluation commands.
