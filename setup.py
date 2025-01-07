from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="starbase_gp",
    version="0.1.0",
    author="",
    author_email="@example.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EpistasisLab/StarBASE",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy==2.0.2",
        "pandas",
        "scikit-learn",
        "ray",
        "matplotlib",
        "typeguard",
        "statsmodels",
        "shap",
        "pyarrow",
        "sortedcontainers",
        "datatable"
    ]
)