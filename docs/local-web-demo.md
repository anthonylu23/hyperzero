# Local Web Demo

This demo runs the universal agent behind a local FastAPI API and a Vite/React
Three.js frontend.

## Setup

```bash
npm install --prefix apps/web
```

The Python dependencies are listed in `pyproject.toml`. The API expects the
promoted universal checkpoint at:

```text
runs/universal_residual_followup_20260528/residual_recovery_lr2e5_seed6603/checkpoints/best_by_eval_score.pt
```

Override it with `HYPERZERO_UNIVERSAL_CHECKPOINT=/path/to/checkpoint.pt` if
needed.

## Run

Start the API:

```bash
make demo-api
```

Start the frontend in another terminal:

```bash
make demo-web
```

Open:

```text
http://127.0.0.1:5173/
```

## Checks

```bash
make test-demo
make build-web
```
