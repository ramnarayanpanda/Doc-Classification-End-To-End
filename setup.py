from setuptools import setup


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="rampanda",
    description="A end to end pacakge for Doc Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ramnarayanpanda/Doc-Classification-End-To-End",
    author_email="rampanda.2597@gmail.com",
    packages=["src"],
    python_requires=">=3.8",
    install_requires=[
        "python>=3.8.0",
        "pandas>=1.4.1",
        "nltk>=3.7",
        "sklearn>=1.0.2",
        "torch>=1.10.2",
        "numpy>=1.22.4",
        "fasttext",
        "unidecode",
        "bs4>=4.10.0",
        "sumy>=0.9.0",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "transformers>=4.17.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "dvc>=2.11.0",
        "mlflow>=1.24.0"
    ]
)