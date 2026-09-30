FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run full pipeline if model artifact does not yet exist
RUN python -m src.pipeline

EXPOSE 8000 8501

# Default command starts FastAPI; override with `streamlit run app/streamlit_app.py` for UI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
