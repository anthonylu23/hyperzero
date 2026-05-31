# Deployment

HyperZero deploys as two services:

- Render runs the FastAPI inference API from the root `Dockerfile`.
- Vercel serves the Vite/React frontend from `apps/web`.
- Both services deploy from the GitHub `main` branch.

Production URLs:

```text
API: https://hyperzero-api.onrender.com
Web: https://hyperzero-web-demo.vercel.app
```

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

Live service settings:

```text
Repository: https://github.com/anthonylu23/hyperzero
Branch: main
Auto deploy: commit
Health check: /health
```

## Frontend: Vercel

The Vercel project is linked to `anthonylu23/hyperzero`, deploys production from
`main`, and uses `apps/web` as the project root. `apps/web/vercel.json` sets:

```text
Build command: npm run build
Output directory: dist
Framework: Vite
```

The Vercel production environment must set:

```text
VITE_API_URL=https://hyperzero-api.onrender.com
```

## Verification

After both services deploy, run:

```bash
python3 scripts/smoke_deployed_demo.py \
  --api-url https://hyperzero-api.onrender.com \
  --web-url https://hyperzero-web-demo.vercel.app
```

A passing smoke confirms the API health endpoint, checkpoint presence/model
load, one human/agent move round trip, and a `200` response from the web app.
