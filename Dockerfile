FROM python:3.12-slim AS base

LABEL maintainer="TensorQuantLib Contributors"
LABEL description="TensorQuantLib — Tensor-Train surrogate pricing engine with autodiff"

WORKDIR /app

# Copy only dependency definitions first for caching
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the library
RUN pip install --no-cache-dir .

# --- Production target ---
FROM base AS production
COPY examples/ ./examples/
COPY benchmarks/ ./benchmarks/
ENTRYPOINT ["python3"]
CMD ["examples/demo_basket_tt.py"]

# --- Development target ---
FROM base AS dev
RUN pip install --no-cache-dir ".[dev]"
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY benchmarks/ ./benchmarks/
ENTRYPOINT ["python3"]
CMD ["-m", "pytest", "tests/", "-v"]
