from setuptools import setup, find_packages

setup(
    name="datapure",
    version="0.1.0",
    description="AI-powered data cleaning for data scientists",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "polars>=0.20.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "anthropic>=0.25.0",
        "ftfy>=6.1.0",
        "rapidfuzz>=3.0.0",
        "chardet>=5.0.0",
        "pyarrow>=14.0.0",
        "openpyxl>=3.1.0",
        "python-dateutil>=2.8.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "ruff", "pre-commit"],
    },
    entry_points={
        "console_scripts": ["datapure=datapure.cli.main:cli"],
    },
)