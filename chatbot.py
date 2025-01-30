from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:3b",
        llm_temperature: float = 0.6,
        qdrant_host: str = "http://localhost:6333",
        db_name: str = "Vector_Database",
    ):

        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_host
        self.db_name = db_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
        )

        
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

        # Initializing Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Initializing the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.db_name
        )


        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

        self.chain_type_kwargs = {"prompt": self.prompt}
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        try:
            response = self.qa.run(query)
            return response 
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
