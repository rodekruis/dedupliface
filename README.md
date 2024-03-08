# dedupliface

Deduplicate kobo submissions using face pictures.

### Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API that checks if face pictures in kobo are duplicate. 

Based on [facenet-pytorch](https://github.com/timesler/facenet-pytorch). Uses [Poetry](https://python-poetry.org/) for dependency management.


### API Usage

See [the docs](https://510-121-dedupliface.azurewebsites.net/docs).

### Run locally

```
cp example.env .env
```
fill in secrets in `.env`
```
pip install poetry
poetry install --no-root
uvicorn main:app --reload
```
