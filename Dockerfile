# Use a slim Python image matching project requirements
FROM python:3.13-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (without installing the project itself)
RUN uv sync --frozen --no-cache

# Copy v2 of the application
COPY api/app/v2 /flowers/api/app/v2
COPY src/ /flowers/src/
# for class labels
COPY data/Oxford-102_Flower_dataset_labels.txt /flowers/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/flowers/
ENV DATA_ROOT=/flowers/data/

# Run the application
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "api.app.v2.main"]
