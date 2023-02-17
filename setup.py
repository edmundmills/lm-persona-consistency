import setuptools


def get_readme():
    """Fetch README from file."""
    with open("README.md", "r") as f:
        return f.read()


setuptools.setup(
    name="persona_consistency",
    version="0.0.1",
    description="Evaluating the Consistency of LM Personas",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8.0",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "openai",
        "tqdm",
        "numpy",
        "matplotlib",
        "pandas",
        "einops"
    ],
    url="https://github.com/edmundmills/lm-persona-consistency]",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
