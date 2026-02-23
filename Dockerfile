# Use a slim Python image matching project requirements
FROM python:3.14-rc-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (without installing the project itself)
RUN uv sync --frozen --no-cache

# Copy the rest of the application
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_MODULE=flowers.api:app
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/flower_model_weights.pth
ENV DATA_ROOT=/app/data

# Run the application
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "flowers.api:app", "--host", "0.0.0.0", "--port", "8000"]
