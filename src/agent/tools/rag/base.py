from typing import Optional

from langchain.embeddings import Embeddings
from langchain.tools import BaseTool
from langchain_community.tools import StructuredTool
from langchain_community.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from pydantic import BaseModel, Field

from src.utils.logger import logger


class DocumentLoaderManager:
    LoaderMap = {
        "txt": TextLoader,
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "doc": Docx2txtLoader,
    }

    @classmethod
    def get_loader(cls, file_name: str) -> BaseLoader:
        file_suffix = file_name.split(".")[-1]
        return cls.LoaderMap[file_suffix]


class RAG:
    def __init__(
        self,
        text_splitter: TextSplitter = None,
        embedding_model: Embeddings = None,
        vector_store: VectorStore = None,
        retriever: BaseRetriever = None,
    ) -> None:
        self.document_loader_manager = DocumentLoaderManager()
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retriever = retriever

    def load_documents(self, file_path: str, lazy_load: bool = False) -> list[Document]:
        document_loader = self.document_loader_manager.get_loader(file_path)
        loader = document_loader(file_path, encoding="utf-8")

        if lazy_load:
            documents = loader.lazy_load()
        else:
            documents = loader.load()

        texts = self.text_splitter.split_documents(documents)

        self.vector_store.add_documents(texts)

    def _perform_retrieve(
        self, query: str, top_k: int = 5, document_id: Optional[str] = None
    ) -> list[Document]:
        """
        Args:
            query (str): 用户的原始问题或提取的关键词，用于语义搜索。
            document_id (str, optional): 指定要搜索的文档ID。如果不指定，将从所有文档中搜索。
            top_k (int, optional): 指定要返回的文档数量。默认为5。

        Returns:
            list[Document]: Document列表。
        """
        logger.info(
            f"RAG retrieve: query={query}, top_k={top_k}, document_id={document_id}"
        )
        documents = self.retriever.invoke(query)
        logger.debug(f"Retrieved {len(documents)} documents")
        return documents[:top_k]

    class SearchInput(BaseModel):
        """检索工具的输入参数 schema。"""

        query: str = Field(description="用户的搜索查询，用于语义检索。")
        top_k: int = Field(default=5, description="返回最相关的结果数量。")
        document_id: Optional[str] = Field(
            default=None, description="可选，限定搜索的文档 ID。"
        )

    def as_tool(self) -> BaseTool:
        """将 RAG 引擎转换为 LangChain 工具。"""
        return StructuredTool.from_function(
            func=self._perform_retrieve,
            name="knowledge_base_search",
            description="在知识库中进行语义搜索，获取与用户问题最相关的信息。",
            args_schema=self.SearchInput,
        )
