FROM python:3.11-slim

WORKDIR /app

# base deps (OpenAGI runtime), then API deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/requirements-api.txt api/requirements-api.txt
RUN pip install --no-cache-dir -r api/requirements-api.txt

COPY . .

# Render injects PORT
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
