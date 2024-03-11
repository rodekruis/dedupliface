# dedupliface

Deduplicate kobo submissions using face pictures.

### Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API that checks if face pictures in kobo are duplicate. 

Based on [facenet-pytorch](https://github.com/timesler/facenet-pytorch). Uses [Poetry](https://python-poetry.org/) for dependency management.


### API Usage

See [the docs](https://510-121-dedupliface.azurewebsites.net/docs).

### Configuration

```sh
cp example.env .env
```

Edit the provided [ENV-variables](./example.env) accordingly.

### Run locally

with Uvicorn (Python web server):
```sh
poetry install
uvicorn main:app --reload
```
with Docker:
```sh
docker compose up --detach
```
