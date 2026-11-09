# Deploying the prospectus bot

Two pieces, two free hosts:

| Piece | Folder | Host |
|---|---|---|
| API (FastAPI, models) | `api/` | Render, free web service (512 MB RAM; Docker, built from this GitHub repo) |
| Website (Next.js) | `web/` | Vercel (free Hobby tier) |

Both need your own accounts; nothing here signs in for you.

The saved models do not use FinBERT scores (the no-text versions won on the validation years), so the API image
leaves out PyTorch and FinBERT and runs in about 320 MB. If a retrained model does use FinBERT, the API loads it
automatically, but the image must then be built with `--build-arg FINBERT=1` and needs about 1.5 GB of memory
(a paid instance).

## 0. Run locally first

```bash
.venv/Scripts/python src/pipeline.py                      # data, features, models, figures
.venv/Scripts/python api/build_examples.py                # demo IPOs for the home page
.venv/Scripts/python -m uvicorn api.app.main:app --port 8000
```

Open http://localhost:8000/docs to try the API. In a second terminal:

```bash
cd web
copy .env.example .env.local
npm run dev
```

Open http://localhost:3000.

## 1. API on Render

1. Sign up at https://render.com with your GitHub account (no card needed for the free plan).
2. **New → Blueprint**, pick this repository. Render reads `render.yaml` and proposes the `prospectus-bot-api`
   web service on the free plan.
3. It asks for `ALLOWED_ORIGINS`: enter your Vercel URL if you have it, or `http://localhost:3000` for now and
   change it in step 2.4 below (service → **Environment**).
4. **Apply**. The first build takes about 5 to 10 minutes. When the service shows **Live**, open
   `https://<service>.onrender.com/health`, then `/docs` to try the API.

Free services sleep after 15 minutes without traffic; the first request after that takes about a minute to wake.
Pushing changes to `api/`, `src/`, `models/` or `data/` on GitHub redeploys automatically.

### Alternative: Hugging Face Spaces (Docker Spaces need a paid plan)

```bash
.venv/Scripts/python deploy/build_space.py   # writes deploy/space/, a separate git repo
cd deploy/space
git init -b main && git lfs install
git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
git add . && git commit -m "Deploy prospectus bot API" && git push -u origin main --force
```

Set `ALLOWED_ORIGINS` under the Space's Settings → Variables.

## 2. Website on Vercel

1. Push the project to GitHub (or just the `web/` folder).
2. At https://vercel.com/new, import the repository and set **Root Directory** to `web`.
3. Add the environment variable `NEXT_PUBLIC_API_URL` = `https://<service>.onrender.com` (no trailing slash).
4. Deploy. Then put the Vercel URL into the Render service's `ALLOWED_ORIGINS` (Environment tab) if you have not yet.

## Updating

After rerunning the pipeline (new data or models): rerun `api/build_examples.py`, then commit and push. Render
rebuilds the API; Vercel rebuilds the website when `web/` changes.
