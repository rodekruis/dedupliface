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


def _get_duplicate_face_ids(vector_store, kobo_client, threshold):
    """ Get IDs of duplicate faces in vector store. """
    submissions = kobo_client.get_kobo_data_bulk()
    duplicate_face_ids = []
    for submission in submissions:
        face1_id = str(submission['_id'])
        face1_vector = np.array(vector_store.client.get_document(face1_id)['content_vector'])
        faces = vector_store.search_face(face1_vector, 4)[1:]  # get top 3 similar faces
        for face in faces:
            face2_id = str(face['id'])
            face2_vector = np.array(face['content_vector'])
            if np.dot(face1_vector, face2_vector) > threshold:
                duplicate_face_ids.append(face1_id)
                duplicate_face_ids.append(face2_id)
    return list(set(duplicate_face_ids))


def _find_duplicates_update_kobo(vector_store, kobo_client, field, value, threshold):
    """ Update Kobo data with duplicate face IDs. """
    duplicate_face_ids = _get_duplicate_face_ids(vector_store, kobo_client, threshold)
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


def add_face_headers(
        koboasset: str = Header(description="ID of the Kobo form (asset)"),
        kobotoken: str = Header(description="your Kobo API token"),
        kobofield: str = Header(description="name of the Kobo field containing the picture")):
    return koboasset, kobotoken, kobofield


@app.post("/add-face")
async def add_face(request: Request, dependencies=Depends(add_face_headers)):
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
    file = kobo_client.get_kobo_attachment(request.headers['kobofield'])
    img = Image.open(BytesIO(file))
    t2_stop = perf_counter()
    logger.info(f"Elapsed time get kobo picture: {float(t2_stop - t2_start)} seconds")
    
    # Detect face and embed it
    t2_start = perf_counter()
    face_img = face_detector(img)
    x_ = face_img.permute(1, 2, 0).int().numpy()
    x_ = trans(x_)
    x_ = x_.unsqueeze(0).to('cpu')
    face_vector = face_identifier(x_).to('cpu').detach().numpy().squeeze(0)
    t2_stop = perf_counter()
    logger.info(f"Elapsed time face detection and embedding: {float(t2_stop - t2_start)} seconds")
    
    # Encrypt face vector

    # Get rotation angle
    hashed_asset = abs(hash(request.headers['koboasset'])) % (10 ** 8)
    rotation_angle = 180. * hashed_asset / (10 ** len(str(hashed_asset)))

    # Get two vectors defining the rotation plane
    n1 = np.array([int(i) for i in list(os.getenv("ROTATION_VECTOR").split(","))]).astype(np.float32)
    n1 /= np.linalg.norm(n1)
    n2 = np.where(n1 == 0., 1., 0.)
    n2 /= np.linalg.norm(n2)

    # Rotate face vector
    rotation_matrix = (np.identity(512) + (np.outer(n2, n1) - np.outer(n1, n2)) * np.sin(rotation_angle) +
                       (np.outer(n1, n1) + np.outer(n2, n2)) * (np.cos(rotation_angle) - 1.))
    face_vector = np.dot(rotation_matrix, face_vector)
    
    # Store face in vector store
    t2_start = perf_counter()
    vector_store = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        store_id=request.headers['koboasset']
    )
    vector_store.add_face(
        face_id=kobo_data['_id'],
        face_vector=face_vector
    )
    t2_stop = perf_counter()
    logger.info(f"Elapsed time store face embedding: {float(t2_stop - t2_start)} seconds")
    
    return JSONResponse(
        status_code=200,
        content={"result": f"Added face of submission {kobo_data['_id']} to vector store."}
    )


class DeduplicatePayload(BaseModel):
    kobofield: str = Field(..., description="""
        Name of the field used to mark duplicates""")
    kobovalue: str = Field(..., description="""
        Value used to mark duplicates (e.g. 'duplicate')""")
    threshold: float = Field(default=0.7, description="""
            How confident you want the model to be
            in order to mark two faces as duplicate,
            on a scale from 0 to 1""")


def deduplicate_headers(
        koboasset: str = Header(description="ID of the Kobo form (asset)"),
        kobotoken: str = Header(description="your Kobo API token")):
    return koboasset, kobotoken


@app.post("/find-duplicate-faces")
async def find_duplicate_faces(payload: DeduplicatePayload, request: Request, background_tasks: BackgroundTasks, dependencies=Depends(deduplicate_headers)):
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
    
    background_tasks.add_task(
        _find_duplicates_update_kobo,
        vector_store,
        kobo_client,
        payload.kobofield,
        payload.kobovalue,
        payload.threshold
    )

    return JSONResponse(
            status_code=202,
            content={"result": f"Duplicates are being checked and marked as '{payload.kobofield}' in field '{payload.kobovalue}'."}
        )
    
    
class Duplicates(BaseModel):
    duplicates: list = Field(..., description="""
        List of IDs of submissions with duplicate faces.""")


def delete_headers(
        koboasset: str = Header(description="ID of the Kobo form (asset)")):
    return koboasset


@app.delete("/delete-faces")
async def delete_faces(request: Request, dependencies=Depends(delete_headers)):
    """Delete faces from vector store."""
    
    vector_store = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        store_id=request.headers['koboasset']
    )
    try:
        vector_store.index_client.delete_index()
    except KeyError:
        raise HTTPException(404, detail=f"No faces found for Kobo asset {request.headers['koboasset']}.")
    return JSONResponse(
        status_code=200,
        content={"result": f"Deleted all faces from vector store."}
    )
    
    
@app.post("/get-duplicates-kobo")
async def get_duplicates_kobo(payload: DeduplicatePayload, request: Request, dependencies=Depends(deduplicate_headers)):
    """Get IDs of duplicates from kobo."""
    
    kobo_client = KoboAPI(
        url="https://kobo.ifrc.org",
        token=request.headers['kobotoken'],
        asset=request.headers['koboasset']
    )
    kobo_data = kobo_client.get_kobo_data_bulk()
    duplicate_face_ids = [k['_id'] for k in kobo_data if k[payload.kobofield] == payload.kobovalue]
    
    response = Duplicates(
        duplicates=duplicate_face_ids
    )
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
