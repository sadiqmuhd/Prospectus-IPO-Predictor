# API image. Build context: the project root (Render, see render.yaml) or the folder made by deploy/build_space.py.
#   docker build -f api/Dockerfile .                       # without FinBERT: the saved models do not use it
#   docker build -f api/Dockerfile --build-arg FINBERT=1 . # with FinBERT (needs about 1 GB more memory)
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
WORKDIR /home/user/app

COPY --chown=user api/requirements.txt api/requirements-finbert.txt api/
RUN pip install --no-cache-dir --user -r api/requirements.txt

# bake FinBERT into the image so the first request is not a 440 MB download
ARG FINBERT=0
RUN if [ "$FINBERT" = "1" ]; then \
      pip install --no-cache-dir --user -r api/requirements-finbert.txt && \
      python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); M.from_pretrained('ProsusAI/finbert')"; \
    fi

COPY --chown=user . .
# Render sets PORT; Hugging Face Spaces expects 7860
EXPOSE 7860
CMD uvicorn api.app.main:app --host 0.0.0.0 --port ${PORT:-7860}
