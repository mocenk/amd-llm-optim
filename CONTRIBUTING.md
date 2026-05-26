# Contributing to amd-llm-optim

Thanks for your interest in contributing! This project aims to make LLM
inference faster, cheaper, and more memory-efficient on AMD ROCm GPUs.
Contributions of all sizes are welcome — bug reports, kernel patches,
benchmark numbers from new hardware, docs fixes.

## Ways to contribute

- **Bug reports** — open an issue with a minimal repro and your ROCm + GPU info
- **New benchmarks** — submit results from MI210 / MI250X / MI300X / Radeon W7900
- **Kernel work** — HIP kernel optimizations, fused ops, autotune configs
- **Quantization** — new schemes (SmoothQuant, SqueezeLLM, etc.)
- **Docs** — clarifications, typo fixes, additional examples

## Development setup

```bash
git clone https://github.com/mocenk/amd-llm-optim.git
cd amd-llm-optim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=optimizer --cov-report=term-missing
```

## Code style

- Black formatting (line length 100)
- Ruff for linting (`ruff check src/ tests/`)
- Type hints on all public functions
- Docstrings in NumPy or Google style

Run the same checks CI runs before pushing:

```bash
ruff check src/ tests/ benchmarks/
mypy --ignore-missing-imports src/optimizer
pytest tests/
```

## Pull request process

1. Fork the repo and create a feature branch (`feat/your-feature`)
2. Make your changes with tests (every public function should have one)
3. Update README / docs if you change user-facing behavior
4. Open a PR against `main` with a clear description and benchmark numbers
   if relevant
5. Sign off your commits (`git commit -s`)

## Hardware-specific contributions

If you're adding support for a new AMD architecture (CDNA3, RDNA4) or a
new ROCm version, please include:

- The exact `rocm-smi` / `hipcc --version` output
- A small reproducible benchmark (`python -m benchmarks.run_benchmark ...`)
- Any kernel autotune configs in `configs/autotune/<arch>.yaml`

## Code of conduct

Be kind, be patient, assume good faith. Discriminatory or harassing
behavior is not tolerated.

## Questions?

Open a [Discussion](https://github.com/mocenk/amd-llm-optim/discussions)
or file an issue with the `question` label.
