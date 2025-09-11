__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import logging
from typing import List, Optional, Dict

# Third-party imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Custom imports
from config import settings

logger = logging.getLogger(__name__)


class ChromaDB:
    """Chroma vector database wrapper"""

    def __init__(self,
                 persist_directory: Optional[str] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "documents"):

        self.persist_directory = persist_directory or settings.chroma_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = self._initialize_vectorstore()

        logger.info(f"ChromaDB initialized with collection: {collection_name}")

    def _initialize_vectorstore(self) -> Chroma:
        """Initialize Chroma vectorstore"""
        try:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}")
            raise

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """Add documents to the vector database"""
        try:
            self.vectorstore.add_documents(documents, ids=ids)
            logger.debug(f"Added {len(documents)} documents to vectorstore")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search"""
        try:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            return []

    def delete_documents(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None) -> None:
        """Delete documents from vectorstore"""
        try:
            self.vectorstore.delete(ids=ids, where=where)
            logger.debug(f"Deleted documents with ids={ids}, where={where}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    def get_documents(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None) -> Dict:
        """Get documents from vectorstore"""
        try:
            return self.vectorstore.get(ids=ids, where=where)
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return {}

    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        """Convert to retriever for use in chains"""
        search_kwargs = search_kwargs or {"k": 10}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def reset_collection(self) -> None:
        """Reset the entire collection"""
        try:
            all_docs = self.vectorstore.get()
            if all_docs and 'ids' in all_docs and all_docs['ids']:
                self.vectorstore.delete(ids=all_docs['ids'])
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise

    def close(self):
        self.vectorstore = None


if __name__ == "__main__":
    from langchain.schema import Document

    print("=== ChromaDB 테스트 ===")
    # ChromaDB 초기화 (임시 디렉토리 사용)
    chroma_db = ChromaDB(persist_directory="./temp_test_db")
    print("ChromaDB 초기화 완료")

    docs = [
        Document(page_content="파이썬은 프로그래밍 언어입니다.", metadata={"topic": "programming"}),
        Document(page_content="머신러닝은 AI의 한 분야입니다.", metadata={"topic": "ai"}),
        Document(page_content="데이터베이스는 정보를 저장합니다.", metadata={"topic": "database"})
    ]

    chroma_db.add_documents(docs)
    print(f"✓ {len(docs)}개 문서 추가")

    # 검색 테스트
    results = chroma_db.similarity_search("프로그래밍", k=2)
    print(f"✓ 검색 결과: {len(results)}개")

    if results:
        print(f"  - 첫 번째 결과: {results[0].page_content}")

    # 점수와 함께 검색
    scored = chroma_db.similarity_search_with_score("AI", k=1)
    if scored:
        doc, score = scored[0]
        print(f"✓ 유사도 점수: {score:.3f} - {doc.page_content}")

    # 전체 문서 개수 확인
    all_docs = chroma_db.get_documents()
    print(f"✓ 저장된 문서: {len(all_docs.get('ids', []))}개")

    print("테스트 완료!")
    chroma_db.close()

    # 테스트 DB 정리
    import shutil
    shutil.rmtree("./temp_test_db", ignore_errors=True)
    print("✓ 임시 파일 정리 완료")
