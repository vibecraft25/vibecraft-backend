__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import logging
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# Custom imports
from services.data_processing.rag import ChromaDB, DocumentProcessor
from schemas.data_schemas import DocumentSearchResult

logger = logging.getLogger(__name__)


class RAGEngine:
    """간소화된 RAG 엔진 - 새로운 컴포넌트 사용"""

    def __init__(self,
                 collection_name: str = "documents",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 persist_directory: Optional[str] = None):

        self.persist_directory = persist_directory or settings.chroma_path

        # 새로운 컴포넌트 초기화
        self.chroma_db = ChromaDB(
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.indexed_files = set()
        logger.info(f"RAG Engine initialized: {persist_directory}")

    def add_document(self, file_path: str) -> bool:
        """단일 문서 추가"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # 이미 인덱싱된 파일인지 확인
            file_id = self._generate_file_id(file_path)
            if file_id in self.indexed_files:
                logger.info(f"Already indexed: {file_path}")
                return True

            # 문서 처리
            chunks = self.document_processor.process_document(file_path)

            if not chunks:
                logger.warning(f"No chunks created for: {file_path}")
                return False

            # 메타데이터 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'chunk_id': f"{file_id}_{i}",
                    'file_id': file_id
                })

            # ChromaDB에 추가
            chunk_ids = [chunk.metadata['chunk_id'] for chunk in chunks]
            self.chroma_db.add_documents(chunks, ids=chunk_ids)

            self.indexed_files.add(file_id)
            logger.info(f"Indexed: {file_path} ({len(chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            return False

    def add_documents_from_directory(self, directory_path: str) -> Dict[str, int]:
        """디렉토리의 모든 지원 문서 추가"""
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return {'success': 0, 'failed': 0}

        success_count = 0
        failed_count = 0

        # 지원되는 파일들 찾기
        supported_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.document_processor.is_supported_file(file_path):
                    supported_files.append(file_path)

        logger.info(f"Found {len(supported_files)} supported files")

        # 각 파일 처리
        for file_path in supported_files:
            if self.add_document(file_path):
                success_count += 1
            else:
                failed_count += 1

        logger.info(f"Indexing complete: {success_count} success, {failed_count} failed")
        return {'success': success_count, 'failed': failed_count}

    def search(self, query: str, k: int = 5) -> List[DocumentSearchResult]:
        """문서 검색"""
        try:
            results = self.chroma_db.similarity_search_with_score(query, k=k)

            search_results = []
            for doc, score in results:
                result = DocumentSearchResult(
                    file_path=doc.metadata.get('file_path', ''),
                    content=doc.page_content,
                    score=1.0 - score,  # 거리를 유사도로 변환
                    metadata=doc.metadata
                )
                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_documents_count(self) -> int:
        """인덱싱된 문서 개수"""
        try:
            all_docs = self.chroma_db.get_documents()
            return len(all_docs.get('ids', []))
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def delete_document(self, file_path: str) -> bool:
        """문서 삭제"""
        try:
            file_id = self._generate_file_id(file_path)

            # 해당 파일의 모든 청크 삭제
            self.chroma_db.delete_documents(where={'file_id': file_id})

            self.indexed_files.discard(file_id)
            logger.info(f"Deleted document: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {file_path}: {e}")
            return False

    def reset(self) -> bool:
        """모든 데이터 초기화"""
        try:
            self.chroma_db.reset_collection()
            self.indexed_files.clear()
            logger.info("RAG Engine reset completed")
            return True
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False

    def as_retriever(self):
        """LangChain retriever로 변환"""
        return self.chroma_db.as_retriever()

    def _generate_file_id(self, file_path: str) -> str:
        """파일 경로 기반 ID 생성"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]


# 싱글톤 인스턴스
from config import settings
rag_engine = RAGEngine(persist_directory=settings.chroma_path)
rag_engine.add_documents_from_directory(f"{settings.data_path}/documents")

if __name__ == '__main__':
    rag_engine.add_documents_from_directory("C:/Users/Administrator/Desktop/Aircok/ffdm-be/storage/documents")
    result = rag_engine.search("Meteorological")
    print(result)
