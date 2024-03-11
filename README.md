# dedupliface 👩🏿👩🏽‍🦱👳🏻

Deduplicate [Kobo](https://www.kobotoolbox.org/) submissions using face pictures.

> [!NOTE]
> ### Terms of Service
> 
> Use of dedupliface is permitted **only**:
> * for **humanitarian programs** involving the registration of people,
> * when **no official proof of identity is available** to people assisted,
> * to **prevent duplicate registrations**, whether caused by error or fraud,
> * with **human validation**, i.e. with one or more humanitarian worker(s) checking duplicates and deciding 
>  inclusion/exclusion in/from the program,
> * in combination with **KoboToolbox**.
> 
> Collection of face pictures and their use in dedupliface must be done in accordance with the [IFRC Data Protection Policy](https://www.ifrc.org/document/IFRC-Data-Protection-Policy).

### Usage

The high-level workflow is:
1. Create a Kobo form with a question of type `Photo`, with which you collect face pictures.
2. Connect the Kobo form with dedupliface using [Kobo REST Services](https://support.kobotoolbox.org/rest_services.html).
3. When a new submission is uploaded to Kobo, an encrypted numerical representation of the face, a.k.a. an _embedding_, 
 is saved in a dedicated _vector database_. The encryption key is unique to the Kobo form.
4. Dedupliface checks which faces in the vector database are duplicate and stores the information in the Kobo database.
6. Delete the encrypted embeddings from the vector database, for extra safety.

For specifics, see [the docs](https://dedupliface.azurewebsites.net/docs).

#### Connect Kobo to dedupliface:

1. Define which question in the Kobo form is used to get face pictures.
2. Define which question in the Kobo form is used to mark duplicates (can be hidden in the form itself).
3. [Register a new Kobo REST Service](https://support.kobotoolbox.org/rest_services.html) and give it a descriptive name.
4. Insert as `Endpoint URL`
   ```
   https://dedupliface.azurewebsites.net/add-face
   ```
6. Add under `Custom HTTP Headers`:
   * In `Name` add `koboasset` and in `Value` the ID of your Kobo form (_asset_)
   * In `Name` add `kobotoken` and in `Value` your Kobo API _token_ (see [how to get one](https://support.kobotoolbox.org/api.html#getting-your-api-token))
   * In `Name` add `kobofield` and in `Value` the name of the question used for face pictures

#### Get duplicates:

1. Upload all submissions to Kobo
2. Make a POST request to
```
https://dedupliface.azurewebsites.net/find-duplicate-faces
```
through the [Swagger UI](https://dedupliface.azurewebsites.net/docs) or whatever tool you prefer. 
   * Specify `koboasset` and `kobotoken` in the headers, as before
   * Specify `kobofield` and `kobovalue` in the request body, where `kobofield` is the name of the question used for marking duplicates and `kobovalue` is the value that marks a duplicate (e.g. `yes`)
3. Your duplicate submissions will now be marked as such in KoboToolbox.

### Technical Specifications

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API that checks if face pictures in Kobo are duplicate. 

Based on [FastAPI](https://fastapi.tiangolo.com/) and [facenet-pytorch](https://github.com/timesler/facenet-pytorch). 
Stores and queries face embeddings with a dedicate vector database, [Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search). 
Uses [Poetry](https://python-poetry.org/) for dependency management.

Encrypts face embeddings with two keys, one global and one unique to each Kobo form.


### Run locally

Configure local environment variables with
```sh
cp example.env .env
```

and edit the provided [ENV-variables](./example.env) accordingly.

Then, with Uvicorn (Python web server):
```sh
poetry install
uvicorn main:app --reload
```
or with Docker:
```sh
docker compose up --detach
```
