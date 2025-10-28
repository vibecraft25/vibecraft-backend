FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy everything
COPY . .

# Install dependencies
RUN uv sync --no-dev || uv pip install -e .

EXPOSE 8080

CMD ["uv", "run", "python", "main.py"]
