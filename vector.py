import os
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,
    CSVLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant

class ManageEmbeddings():
    def __init__(self,
                 model_name="BAAI/bge-small-en",
                 device: str = "cpu",
                 encode_kwargs: dict = {"normalize_embeddings": True},  # Fixed typo
                 qdrant_host: str = "http://localhost:6333/",
                 db_name: str = "Vector_Database",
                 ):

        self.model_name = model_name
        self.device = device
        self.qdrant_host = qdrant_host
        self.encode_kwargs = encode_kwargs
        self.db_name = db_name
        self.client = self.connect_to_qdrant()

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def connect_to_qdrant(self):
        from qdrant_client import QdrantClient
        return QdrantClient(self.qdrant_host)

    def clear_existing_embeddings(self):
        try:
            self.client.delete_collection(self.db_name)  # Delete existing collection
            self.client.recreate_collection(self.db_name, vectors_config={"size": 384, "distance": "Cosine"})
            return "Old embeddings cleared successfully!"
        except Exception as e:
            return f"⚠️ Error clearing embeddings: {e}"

    def embed(self, file_path: str):

        self.clear_existing_embeddings()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Mapping of file types to respective loaders
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".csv": CSVLoader
        }

        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in loaders:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported formats: {list(loaders.keys())}")

        loader = loaders[file_extension](file_path)

        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the file.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250).split_documents(docs)

        if not text_splitter:
            raise ValueError("No text chunks were created from the document.")

        try:
            qdrant_db = Qdrant.from_documents(
                text_splitter,
                self.embeddings,
                url=self.qdrant_host,
                prefer_grpc=False,
                collection_name=self.db_name
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "Vector DB Successfully Created and Stored in Qdrant!"
    
