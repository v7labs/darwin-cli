import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="darwin-py",
    version="0.4.2",
    author="V7",
    author_email="info@v7labs.com",
    description="Library and command line interface for darwin.v7labs.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/v7labs/darwin-py",
    install_requires=[
        "argcomplete",
        "docutils",
        "factory_boy",
        "humanize",
        "pyyaml>=5.1",
        "requests",
        "scikit-learn",
        "sh",
        "tqdm",
    ],
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["darwin=darwin.cli:main"]},
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    python_requires=">=3.6",
)
