"""
Hybrid Retriever: cimbining BM25 and Vector searh for optimal document retrieval
BM25 retrieves highly precise keywords matches, while vector retrieval finds related concepts
"""
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self) -> None:
        """Initialize the retriever builder with embeddings"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}, # Change to 'cuda' if you are running on a GPU
            encode_kwargs={"normalize_embeddings": True}
        )

    def build_hybrid_retriever(self, docs):
        """Build hybrod retriever using BM25 retriever and vector-based retriever"""
        try:
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory = settings.CHROMA_DB_PATH
            )
            logger.info("Vector store created succesfully")

            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriver succesfully")

            vector_retriever = vector_store.as_retriever(search_kwargs={"k":settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created succesfully")
            
            # Combine retriever into hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights = settings.HYBRID_RETRIEVER_WEIGHTS
            )
            logger.info("Hybrid retriever created successfully")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")