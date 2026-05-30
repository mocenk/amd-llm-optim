"""Package configuration for amd-llm-optim."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def _read_requirements() -> list[str]:
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return []
    lines = req_file.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="amd-llm-optim",
    version="0.3.0",
    description="LLM inference optimization toolkit for AMD ROCm GPUs",
    long_description=README,
    long_description_content_type="text/markdown",
    author="AMD LLM Optim Contributors",
    license="MIT",
    url="https://github.com/example/amd-llm-optim",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=_read_requirements(),
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff>=0.1.0", "mypy>=1.7.0"],
    },
    entry_points={
        "console_scripts": [
            "amd-llm-bench=benchmarks.run_benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
