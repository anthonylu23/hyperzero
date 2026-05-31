# Deployment

HyperZero deploys as two services:

- Render runs the FastAPI inference API from the root `Dockerfile`.
- Vercel serves the Vite/React frontend from `apps/web`.

## Backend: Render

The Render service is configured in `render.yaml` as `hyperzero-api`.
It builds a Docker image from the repository root and runs:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT:-10000}
```

The Docker image includes only the promoted universal checkpoint:

```text
runs/universal_residual_followup_20260528/residual_recovery_lr2e5_seed6603/checkpoints/best_by_eval_score.pt
```

Expected checkpoint SHA-256:

```text
639f7d0241740ee09f080f3e46df516bcda9b4d6da20a02a999e4737e4f7ed68
```

Production environment:

```text
HYPERZERO_DEVICE=cpu
HYPERZERO_PRELOAD_MODEL=1
HYPERZERO_ALLOWED_ORIGIN_REGEX=^https://[a-z0-9-]+\.vercel\.app$
```

## Frontend: Vercel

The Vercel project root is `apps/web`, with `apps/web/vercel.json` setting:

```text
Build command: npm run build
Output directory: dist
Framework: Vite
```

Set this Vercel environment variable after the Render API URL is known:

```text
VITE_API_URL=https://<render-service>.onrender.com
```

## Verification

After both services deploy:

```bash
python3 scripts/smoke_deployed_demo.py \
  --api-url https://<render-service>.onrender.com \
  --web-url https://<vercel-service>.vercel.app
```
