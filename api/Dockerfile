# Hugging Face Spaces (Docker SDK). Build context: the folder made by deploy/build_space.py
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
WORKDIR /home/user/app

COPY --chown=user api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir --user -r api/requirements.txt

# bake FinBERT into the image so the first request is not a 440 MB download
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); M.from_pretrained('ProsusAI/finbert')"

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
