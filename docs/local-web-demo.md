# Local Web Demo

This demo runs the universal agent behind a local FastAPI API and a Vite/React
Three.js frontend.

## Setup

From the repository root, install the frontend dependencies:

```bash
npm install --prefix apps/web
```

For a fresh Python environment, install the direct runtime/test dependencies
used by the local API:

```bash
python3 -m pip install fastapi numpy torch "uvicorn[standard]" pytest ruff
```

The API expects the promoted universal checkpoint at:

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

The hosted version runs the same API/web flow at
`https://hyperzero-web-demo.vercel.app`.

## Checks

```bash
make test-demo
make build-web
python3 scripts/smoke_deployed_demo.py \
  --api-url https://hyperzero-api.onrender.com \
  --web-url https://hyperzero-web-demo.vercel.app
```
