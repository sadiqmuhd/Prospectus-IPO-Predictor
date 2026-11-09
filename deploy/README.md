# Deploying the prospectus bot

Two pieces, two hosts:

| Piece | Folder | Host |
|---|---|---|
| API (FastAPI, models) | `api/` | Railway (Docker, built from this GitHub repo; settings in `railway.json`) |
| Website (Next.js) | `web/` | Vercel (free Hobby tier) |

Both need your own accounts; nothing here signs in for you.

The saved models do not use FinBERT scores (the no-text versions won on the validation years), so the API image
leaves out PyTorch and FinBERT and runs in about 320 MB. If a retrained model does use FinBERT, the API loads it
automatically, but the image must then be built with `--build-arg FINBERT=1` and needs about 1.5 GB of memory
(on Railway, add the service variable `FINBERT=1`; Railway passes it to the Docker build).

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

## 1. API on Railway

1. At https://railway.com/new choose **Deploy from GitHub repo** and pick this repository. Railway reads
   `railway.json`: it builds `api/Dockerfile` from the project root and checks `/health` before going live.
2. In the service's **Variables**, add `ALLOWED_ORIGINS` (your Vercel URL, or `http://localhost:3000` until you
   have it) and optionally `ALLOWED_ORIGIN_REGEX` = `https://.*\.vercel\.app` for preview deployments.
3. In **Settings → Networking**, click **Generate Domain** to get a public `https://<name>.up.railway.app` URL.
4. The first build takes about 5 minutes. When the deployment is **Active**, open `/health`, then `/docs`.

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
3. Add the environment variable `NEXT_PUBLIC_API_URL` = `https://<name>.up.railway.app` (no trailing slash).
4. Deploy. Then put the Vercel URL into the Railway service's `ALLOWED_ORIGINS` (Variables tab) if you have not yet.

## Updating

After rerunning the pipeline (new data or models): rerun `api/build_examples.py`, then commit and push. Railway
rebuilds the API; Vercel rebuilds the website when `web/` changes.
