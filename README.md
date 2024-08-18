# Welcome to the SCHEMATIC repository

This repository contains the code used to score and prioritize interactions
identified in the paper "A multi-lineage screen identifies actionable
synthetic-lethal interactions in human cancers" (link to reference will be added
after the article is published.)

## Environment

This package uses [`uv`](https://docs.astral.sh/uv/) to manage project
dependencies. `uv` is not required to install the dependencies as the project
dependencies are listed in the `pyproject.toml` file. However, to reproduce
exactly the environment using the `uv.lock` file, `uv` needs to be installed.

To reproduce the project environment, simply run 

```bash
uv sync
```

To install the package and their dependencies, run 

```bash

pip install .
```

## Running the code

The code is organized in two parts: the library code (in `src`) folder and
scripts (in `scripts`) folder. The scripts use the library code, so the package
needs to be installed before running the code.
