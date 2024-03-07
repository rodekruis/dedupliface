# dedupliface

Chat with HIA.

## Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API to deduplicate kobo submissions using face pictures. Based on [facenet-pytorch](https://github.com/timesler/facenet-pytorch).

## API Usage

See [the docs](https://510-121-dedupliface.azurewebsites.net/docs).

### Run locally

```
cp example.env .env
pip install poetry
poetry install --no-root
uvicorn main:app --reload
```
