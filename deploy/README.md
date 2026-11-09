# Deploying the prospectus bot

Two pieces, two free hosts:

| Piece | Folder | Host |
|---|---|---|
| API (FastAPI, FinBERT, models) | `api/` | Hugging Face Spaces, Docker SDK (free CPU tier: 2 vCPU, 16 GB RAM) |
| Website (Next.js) | `web/` | Vercel (free Hobby tier) |

Both need your own accounts; nothing here signs in for you.

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

## 1. API on Hugging Face Spaces

1. Create a Space at https://huggingface.co/new-space: pick **Docker** as the SDK, blank template, CPU basic (free).
2. Build the Space folder (code, models and the small data files the API needs; about 60 MB):

   ```bash
   .venv/Scripts/python deploy/build_space.py
   ```

3. Push `deploy/space/` to the Space's git repository (it is a separate repo; install Git LFS first, since the
   model and data files are tracked with LFS):

   ```bash
   cd deploy/space
   git init
   git lfs install
   git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
   git add .
   git commit -m "Deploy prospectus bot API"
   git push -u origin main
   ```

   Hugging Face asks for an access token as the password (Settings → Access Tokens, "write" scope).
4. In the Space's **Settings → Variables**, add `ALLOWED_ORIGINS` = your Vercel URL (for example
   `https://prospectus-bot.vercel.app`). Optionally add `ALLOWED_ORIGIN_REGEX` = `https://.*\.vercel\.app` to allow
   preview deployments too.
5. The first build takes about 10 minutes (it installs PyTorch and bakes FinBERT into the image). When it shows
   "Running", check `https://<your-username>-<space-name>.hf.space/health`.

Free Spaces sleep after 48 hours without traffic; the first request after that takes a minute or two to wake.

## 2. Website on Vercel

1. Push the project to GitHub (or just the `web/` folder).
2. At https://vercel.com/new, import the repository and set **Root Directory** to `web`.
3. Add the environment variable `NEXT_PUBLIC_API_URL` = `https://<your-username>-<space-name>.hf.space`.
4. Deploy. Then put the Vercel URL into the Space's `ALLOWED_ORIGINS` (step 1.4) if you have not yet.

## Updating

After rerunning the pipeline (new data or models): rerun `api/build_examples.py` and `deploy/build_space.py`, then
commit and push `deploy/space/` again. The website only needs redeploying when `web/` changes.
