from typing import List

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools.base import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent.tools.rag.base import RAG
from src.config import settings
from src.utils.logger import logger


class RAGManager:
    def __init__(self):
        self.rag = self.create_rag()
        self.tools = self.get_all_rag_tools()

    def create_rag(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        embedding_model = DashScopeEmbeddings(
            model=settings.rag_embedding_model,
            dashscope_api_key=settings.dashscope_llm_api_key,
        )
        vector_store = Chroma(
            collection_name=settings.rag_collection_name,
            embedding_function=embedding_model,
            persist_directory=settings.rag_persist_directory,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": settings.rag_top_k})
        rag = RAG(
            text_splitter=splitter,
            embedding_model=embedding_model,
            vector_store=vector_store,
            retriever=retriever,
        )
        return rag

    def get_all_rag_tools(self) -> List[BaseTool]:
        return [self.rag.as_tool()]

    def load_file(self, file_path: str) -> None:
        logger.info(f"Loading file into RAG: {file_path}")
        self.rag.load_documents(file_path)

    def upload_file(self, file_path: str) -> None:
        self.load_file(file_path)
