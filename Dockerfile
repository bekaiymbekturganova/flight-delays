FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# Mount a trained model at /app/artifacts (see README).
EXPOSE 8000
CMD ["uvicorn", "flightdelays.api:app", "--host", "0.0.0.0", "--port", "8000"]
