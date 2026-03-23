# Contributing to TensorQuantLib

Thank you for your interest in contributing! This guide will help you get
started.

## Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/tensorquantlib.git
cd tensorquantlib

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify the installation
python3 -m pytest
```

## Project Layout

```
src/tensorquantlib/
├── core/          # Tensor class, reverse-mode autodiff, second-order AD
├── finance/       # BS, Heston, American, exotics, SABR/SVI, rates, FX, credit, risk
├── tt/            # TT-SVD decomposition, TT ops, TTSurrogate, TT pricing
├── backtest/      # Backtesting engine, strategies, performance metrics
├── data/          # Market data integration (Yahoo Finance)
├── utils/         # Numerical gradient checking
└── viz/           # Matplotlib plotting utilities
tests/             # pytest test suite (588 tests)
examples/          # Runnable demo scripts
benchmarks/        # Performance benchmarks
docs/              # Sphinx documentation source
```

## Coding Standards

### Style

- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/) — configuration
  lives in `pyproject.toml`.
- **Type checking**: [mypy](https://mypy-lang.org/) in strict mode.
- **Line length**: 88 characters (Black-compatible).
- Run both before committing:

  ```bash
  ruff check src/ tests/
  mypy src/
  ```

### Docstrings

Use Google-style docstrings on all public functions and classes:

```python
def tt_add(cores_a: List[np.ndarray], cores_b: List[np.ndarray]) -> List[np.ndarray]:
    """Add two TT tensors: C = A + B.

    The result has TT-ranks that are the sum of the input ranks.

    Args:
        cores_a: TT-cores of tensor A.
        cores_b: TT-cores of tensor B.

    Returns:
        TT-cores of A + B.

    Raises:
        ValueError: If tensors have different numbers of cores.
    """
```

### Tests

- Every new feature needs tests in `tests/`.
- Use descriptive test names: `test_add_3d_tensors_matches_numpy`.
- Prefer `np.testing.assert_allclose` for numerical comparisons.
- Run the full suite:

  ```bash
  python3 -m pytest -v
  ```

## Making Changes

### Branching

1. Fork the repo and create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Make small, focused commits with clear messages.
3. Keep each PR addressing one concern.

### Pull Request Checklist

Before opening a PR, please ensure:

- [ ] All tests pass: `python3 -m pytest`
- [ ] Linter is clean: `ruff check src/ tests/`
- [ ] Type checker passes: `mypy src/`
- [ ] New code has tests with good coverage
- [ ] Docstrings added for public APIs
- [ ] `CHANGELOG.md` updated (if user-facing change)

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add tt_add and tt_scale arithmetic operations
fix: correct dimension handling in tt_round QR sweep
docs: update README with basket option example
test: add gradient checks for norm_cdf operation
```

## Reporting Issues

- Check existing issues first to avoid duplicates.
- Include a minimal reproducible example.
- Mention your Python version, OS, and NumPy version.

## Adding a New TT Operation

1. Add the function to `src/tensorquantlib/tt/ops.py`.
2. Export it in `src/tensorquantlib/tt/__init__.py`.
3. Export it in `src/tensorquantlib/__init__.py`.
4. Write tests in `tests/test_tt_ops.py` or a dedicated test file.
5. Add a docstring with Args/Returns/Raises sections.

## Adding a New Financial Model

1. Create a module in `src/tensorquantlib/finance/`.
2. Support both NumPy and Tensor versions if autograd is desired.
3. Export in `src/tensorquantlib/finance/__init__.py`.
4. Write tests comparing against known analytical results.

## Code of Conduct

Be respectful, constructive, and inclusive.  We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
