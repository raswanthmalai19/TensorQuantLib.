# Deployment Guide

This guide covers deploying TensorQuantLib as a production library — from
local installation to Docker containerization to cloud deployment.

---

## Table of Contents

1. [Local Installation](#local-installation)
2. [Docker](#docker)
3. [PyPI Publishing](#pypi-publishing)
4. [Cloud Deployment](#cloud-deployment)
5. [Batch Pricing Patterns](#batch-pricing-patterns)
6. [Configuration Reference](#configuration-reference)

---

## Local Installation

### From Source (Development)

```bash
git clone https://github.com/your-org/tensorquantlib.git
cd tensorquantlib
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Verify
python3 -m pytest tests/ -q
python3 -c "import tensorquantlib; print(tensorquantlib.__version__)"
```

### From Wheel (Production)

```bash
# Build the wheel
pip install build
python -m build

# Install anywhere
pip install dist/tensorquantlib-0.1.0-py3-none-any.whl
```

---

## Docker

### Minimal Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (none needed for TensorQuantLib)
# RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Copy application code
COPY scripts/ ./scripts/

ENTRYPOINT ["python3"]
CMD ["scripts/price_basket.py"]
```

### Docker Compose (with Jupyter)

```yaml
version: "3.8"
services:
  pricing:
    build: .
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - TENSORQUANTLIB_LOG_LEVEL=INFO

  notebook:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
    command: >
      jupyter lab --ip=0.0.0.0 --no-browser --allow-root
      --NotebookApp.token=''
    depends_on:
      - pricing
```

### Build & Run

```bash
# Build
docker build -t tensorquantlib:latest .

# Run pricing script
docker run --rm -v $(pwd)/results:/app/results tensorquantlib:latest scripts/price_basket.py

# Run tests inside container
docker run --rm tensorquantlib:latest -m pytest tests/ -q
```

---

## PyPI Publishing

The repository includes a GitHub Actions workflow for automated PyPI
publishing via trusted publishing (no API tokens needed).

### Setup (One-Time)

1. Go to [pypi.org](https://pypi.org) → Your Account → Publishing
2. Add a new "pending publisher":
   - **PyPI project name**: `tensorquantlib`
   - **Owner**: your GitHub username/org
   - **Repository name**: `tensorquantlib`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
3. In your GitHub repo → Settings → Environments → Create `pypi` environment
4. Optionally add required reviewers for release approval

### Publishing a Release

```bash
# 1. Update version in pyproject.toml and src/tensorquantlib/__init__.py
# 2. Update CHANGELOG.md
# 3. Commit and tag
git add -A
git commit -m "release: v0.1.0"
git tag v0.1.0
git push origin main --tags

# 4. Create a GitHub Release from the tag
#    → This triggers .github/workflows/publish.yml automatically
```

### Manual Publishing (TestPyPI)

```bash
pip install build twine

python -m build
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test the install
pip install --index-url https://test.pypi.org/simple/ tensorquantlib

# Upload to production PyPI
twine upload dist/*
```

---

## Cloud Deployment

### AWS Lambda (Serverless Pricing API)

```python
# lambda_handler.py
import json
from tensorquantlib import TTSurrogate

# Build surrogate at cold-start (cached across invocations)
SURROGATE = TTSurrogate.from_basket_analytic(
    S0_ranges=[(80, 120)] * 3,
    K=100, T=1.0, r=0.05,
    sigma=[0.2, 0.25, 0.3],
    weights=[1/3, 1/3, 1/3],
    n_points=30, eps=1e-4,
)

def handler(event, context):
    spots = event.get("spots", [100.0, 100.0, 100.0])
    price = SURROGATE.evaluate(spots)
    greeks = SURROGATE.greeks(spots)
    return {
        "statusCode": 200,
        "body": json.dumps({
            "price": round(price, 6),
            "delta": [round(d, 6) for d in greeks["delta"]],
            "gamma": [round(g, 6) for g in greeks["gamma"]],
        }),
    }
```

**Deployment**:
```bash
# Package as a Lambda layer
pip install tensorquantlib -t python/
zip -r tensorquantlib-layer.zip python/

# Deploy via AWS CLI
aws lambda publish-layer-version \
    --layer-name tensorquantlib \
    --zip-file fileb://tensorquantlib-layer.zip \
    --compatible-runtimes python3.11 python3.12
```

### Google Cloud Run (Container)

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/tensorquantlib

# Deploy
gcloud run deploy tensorquantlib-pricing \
    --image gcr.io/YOUR_PROJECT/tensorquantlib \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 10 \
    --allow-unauthenticated
```

### Azure Container Instances

```bash
az container create \
    --resource-group myResourceGroup \
    --name tensorquantlib-pricing \
    --image tensorquantlib:latest \
    --cpu 1 --memory 1 \
    --restart-policy Never
```

---

## Batch Pricing Patterns

### Pattern 1: Pre-compute and Cache

```python
import numpy as np
import pickle
from tensorquantlib import TTSurrogate

# Build once
surr = TTSurrogate.from_basket_analytic(
    S0_ranges=[(80, 120)] * 3,
    K=100, T=1.0, r=0.05,
    sigma=[0.2, 0.25, 0.3],
    weights=[1/3, 1/3, 1/3],
    n_points=30, eps=1e-4,
)

# Serialize for reuse
with open("surrogate_3asset.pkl", "wb") as f:
    pickle.dump(surr, f)

# Load later
with open("surrogate_3asset.pkl", "rb") as f:
    surr = pickle.load(f)
```

### Pattern 2: Batch Evaluation

```python
# Evaluate thousands of scenarios
scenarios = np.random.uniform(80, 120, size=(10_000, 3))
prices = [surr.evaluate(s.tolist()) for s in scenarios]
```

### Pattern 3: Greek Surface Generation

```python
import numpy as np

# Generate a delta surface over a 2D slice
spots_1 = np.linspace(80, 120, 50)
spots_2 = np.linspace(80, 120, 50)
deltas = np.zeros((50, 50))

for i, s1 in enumerate(spots_1):
    for j, s2 in enumerate(spots_2):
        g = surr.greeks([s1, s2, 100.0])
        deltas[i, j] = g["delta"][0]  # Delta w.r.t. asset 1
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_points` | 30 | Grid resolution per axis |
| `eps` | 1e-4 | TT-SVD truncation tolerance |
| `max_rank` | None | Maximum TT-rank (None = adaptive) |
| `n_paths` | 100,000 | Monte Carlo paths (MC constructor) |
| `option_type` | "call" | "call" or "put" |

### Resource Estimates

| Assets | n_points | RAM (peak) | Build Time | Surrogate Size |
|--------|----------|-----------|-----------|----------------|
| 2 | 30 | ~2 MB | <1s | ~3 KB |
| 3 | 30 | ~10 MB | <1s | ~28 KB |
| 4 | 20 | ~50 MB | ~2s | ~91 KB |
| 5 | 15 | ~200 MB | ~10s | ~142 KB |
