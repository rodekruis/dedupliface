import copy
import numpy as np
from typing import List
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile
)
VECTOR_SEARCH_DIMENSIONS = 512


class VectorStore:
    """
    Vector storage for face embeddings
    """
    def __init__(
        self,
        store_path: str,
        store_password: str = None,
        store_id: str = "face_embeddings"
    ):
        self.store_id = store_id
        self.store_password = store_password
        self.store_path = store_path
        self.index_client = self._set_index_client()
        self.client = self._set_client()
        
    def _set_index_client(self):
        return SearchIndexClient(
            self.store_path,
            AzureKeyCredential(self.store_password)
        )
        
    def _create_azuresearch_index(self):
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=VECTOR_SEARCH_DIMENSIONS,
                vector_search_profile_name="myHnswProfile",
            )
        ]
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ]
        )
        self.index_client.create_index(SearchIndex(name=self.store_id, fields=fields, vector_search=vector_search))

    def _set_client(self):
        """Sets the vector store client"""
        indexes = [index.name for index in self.index_client.list_indexes()]
        if self.store_id not in indexes:
            self._create_azuresearch_index()
        return SearchClient(
            self.store_path,
            index_name=self.store_id,
            credential=AzureKeyCredential(self.store_password)
        )
        
    def add_face(self, face_id: str, face_vector: np.array):
        """Add new face embedding to the vector store"""
        doc = {
            "id": face_id,
            "content_vector": face_vector.tolist()
        }
        self.client.upload_documents(documents=[doc])
        
    def search_face(self, face_vector: np.array, k: int):
        """Search for a face embedding in the vector store"""
        queries = [VectorizedQuery(
            vector=face_vector,
            k_nearest_neighbors=k,
            fields="content_vector",
            exhaustive=True
        )]
        results = self.client.search(vector_queries=queries)
        return [face for face in results]
