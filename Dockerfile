# Use a slim Python base image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

# Install Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy project metadata first to leverage Docker layer caching
COPY pyproject.toml ./

# Install dependencies (no venv for simplicity in container)
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--app-dir", "src"]

