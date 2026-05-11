from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="datapure",
    version="0.1.0",
    description="AI-powered data cleaning library for data scientists",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Basil Anil",
    author_email="basilanil8@gmail.com",
    url="https://github.com/Blitzhac/datapure",
    license="MIT",
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
        "dev": ["pytest", "pytest-cov", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "datapure=datapure.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "data cleaning", "data science", "pandas", "polars",
        "AI", "machine learning", "data quality",
    ],
)