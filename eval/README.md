# Evaluation

## 1. API configuration

Create `.env` in the repository root and fill in the required API keys:

```bash
cp .env.example .env
```

## 2. Single-GPU evaluation

Run both benchmarks from the repository root:

```bash
bash scripts/run_videomme.sh
bash scripts/run_m3bench.sh
```

Both commands use GPU 0 by default and load `.env` automatically. Results are
written to `outputs/videomme/` and `outputs/m3bench/`.
