# exp--pytorch-transformers

This is a playground repository where I am experimenting with automatic extractive summarization with Transformers.

## Development

As a pre-requisite, this project requires [Conda](https://docs.conda.io/en/latest/) and [Poetry](https://python-poetry.org/). To setup the workspace, do the following:

```bash
$ conda create -p ./env python=3.8
$ conda activate ./env
$ conda install jupyter cudatoolkit=11.1 -c pytorch -c nvidia
$ poetry install
```