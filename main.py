from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
from torchvision import datasets, transforms
import numpy as np
from time import perf_counter
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi import Depends, FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from src.vector_store import VectorStore
from src.kobo_api_client import KoboAPI
import os
import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
from dotenv import load_dotenv
load_dotenv()
SIMILARITY_THRESHOLD = 0.9

# load environment variables
if "PORT" not in os.environ.keys():
    port = 8000
else:
    port = os.environ["PORT"]
    
description = """
Deduplicate kobo submissions using face pictures 👩🏽‍🦱👳🏻

Built with love by [NLRC 510](https://www.510.global/). See
[the project on GitHub](https://github.com/rodekruis/dedupliface) or [contact us](mailto:support@510.global).
"""

# initialize FastAPI
app = FastAPI(
    title="dedupliface",
    description=description,
    version="0.0.1",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# initialize face detection and recognition models
face_detector = MTCNN(margin=100, post_process=False)
face_identifier = InceptionResnetV1(
    classify=False,
    pretrained='vggface2'
).to('cpu')
face_identifier.eval()
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])


def _get_duplicate_face_ids(vector_store, kobo_client):
    submissions = kobo_client.get_kobo_data_bulk()
    duplicate_face_ids = []
    for submission in submissions:
        face1_id = submission['id']
        face1_vector = vector_store.client.get_document(face1_id)
        # get top 3 similar faces
        faces = vector_store.search_face(face1_vector, 3)
        for face in faces:
            if face['@search.score'] > SIMILARITY_THRESHOLD:
                duplicate_face_ids.append(face1_id)
                duplicate_face_ids.append(face['id'])
    return list(set(duplicate_face_ids))


def _update_kobo(vector_store, kobo_client, field, value):
    duplicate_face_ids = _get_duplicate_face_ids(vector_store, kobo_client)
    try:
        kobo_client.update_kobo_data_bulk(
            duplicate_face_ids,
            field,
            value
        )
    except RuntimeError as e:
        pass  # TBI


@app.get("/", include_in_schema=False)
async def docs_redirect():
    """Redirect base URL to docs."""
    return RedirectResponse(url='/docs')


def required_kobo_headers(
        koboasset: str = Header(),
        kobotoken: str = Header()):
    return koboasset, kobotoken

class AddFacePayload(BaseModel):
    picturefield: str = Field(..., description="""
        Name of the kobo field containing the picture""")


@app.post("/add-face")
async def add_face(payload: AddFacePayload, request: Request, dependencies=Depends(required_kobo_headers)):
    """Extract face from kobo picture, encrypt, and add to vector store."""
    
    kobo_data = await request.json()
    
    # Get image from Kobo
    t2_start = perf_counter()
    kobo_client = KoboAPI(
        url="https://kobo.ifrc.org",
        token=request.headers['kobotoken'],
        asset=request.headers['koboasset'],
        submission=kobo_data
    )
    file = kobo_client.get_kobo_attachment(payload.picturefield)
    img = Image.open(BytesIO(file))
    t2_stop = perf_counter()
    logger.info(f"Elapsed time get kobo picture: {float(t2_stop - t2_start)} seconds")
    
    # Detect face and embed it
    t2_start = perf_counter()
    face_img = face_detector(img)
    x_ = face_img.permute(1, 2, 0).int().numpy()
    x_ = trans(x_)
    x_ = x_.unsqueeze(0).to('cpu')
    face_vector = face_identifier(x_).to('cpu').detach().numpy()
    t2_stop = perf_counter()
    logger.info(f"Elapsed time face detection and embedding: {float(t2_stop - t2_start)} seconds")
    
    # Store face in vector store
    t2_start = perf_counter()
    vector_store = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        store_id=request.headers['koboasset']
    )
    vector_store.add_face(
        face_id=kobo_data['id'],
        face_vector=face_vector
    )
    t2_stop = perf_counter()
    logger.info(f"Elapsed time store face embedding: {float(t2_stop - t2_start)} seconds")
    
    return JSONResponse(
        status_code=200,
        content={"result": f"Added face of submission {kobo_data['id']} to vector store."}
    )


class DeduplicatePayload(BaseModel):
    duplicatefield: str = Field(..., description="""
        Name of the field used to mark duplicates""")
    duplicatevalue: str = Field(..., description="""
        Value used to mark duplicates""")


@app.post("/find-duplicate-faces")
async def find_duplicate_faces(payload: DeduplicatePayload, request: Request, background_tasks: BackgroundTasks, dependencies=Depends(required_kobo_headers)):
    """Find duplicate faces in vector store and update kobo accordingly."""
    
    vector_store = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        store_id=request.headers['koboasset']
    )
    kobo_client = KoboAPI(
        url="https://kobo.ifrc.org",
        token=request.headers['kobotoken'],
        asset=request.headers['koboasset']
    )
    
    background_tasks.add_task(_update_kobo, vector_store, kobo_client, payload.duplicatefield, payload.duplicatevalue)

    return JSONResponse(
            status_code=202,
            content={"result": f"Duplicates are being checked and marked as '{payload.duplicatevalue}' in field '{payload.duplicatefield}'."}
        )
    
    
class Duplicates(BaseModel):
    duplicates: list = Field(..., description="""
        List of IDs of submissions with duplicate faces.""")
    
    
@app.post("/get-duplicates-kobo")
async def get_duplicates_kobo(payload: DeduplicatePayload, request: Request, dependencies=Depends(required_kobo_headers)):
    """Get IDs of duplicates from kobo."""
    
    kobo_client = KoboAPI(
        url="https://kobo.ifrc.org",
        token=request.headers['kobotoken'],
        asset=request.headers['koboasset']
    )
    kobo_data = kobo_client.get_kobo_data_bulk()
    duplicate_face_ids = [k['id'] for k in kobo_data if k[payload.duplicatefield] == payload.duplicatevalue]
    
    response = Duplicates(
        duplicates=duplicate_face_ids
    )
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
