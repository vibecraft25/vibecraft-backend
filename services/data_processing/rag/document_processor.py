__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import logging
from pathlib import Path
from typing import List, Optional, Union

# Third-party imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing and text chunking utility"""

    SUPPORTED_EXTENSIONS = ('.pdf', '.txt', '.xlsx', '.xls', '.md', '.markdown')

    def __init__(self,
                 chunk_size: int = 600,
                 chunk_overlap: int = 0,
                 separators: Optional[List[str]] = None):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

        logger.info(f"DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file type is supported"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def get_loader_for_file(self, file_path: Union[str, Path]):
        """Get appropriate document loader for file type"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                return PyPDFLoader(str(file_path))
            elif suffix == '.txt':
                return self._create_text_loader(str(file_path))
            elif suffix in ('.xlsx', '.xls'):
                return UnstructuredExcelLoader(str(file_path))
            elif suffix in ('.md', '.markdown'):
                return UnstructuredMarkdownLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as e:
            logger.error(f"Failed to create loader for {file_path}: {e}")
            raise

    def _create_text_loader(self, file_path: str):
        """Create TextLoader with proper encoding handling"""
        # 시도할 인코딩 순서
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']

        for encoding in encodings:
            try:
                # 파일을 해당 인코딩으로 읽어보기 테스트
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(100)  # 처음 100자만 읽어서 테스트

                # 성공하면 해당 인코딩으로 TextLoader 생성
                return TextLoader(file_path, encoding=encoding)

            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error testing encoding {encoding} for {file_path}: {e}")
                continue

        # 모든 인코딩 실패 시 기본값으로 시도
        logger.warning(f"Could not detect encoding for {file_path}, using utf-8 with error handling")
        return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)

    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document from file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            loader = self.get_loader_for_file(file_path)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.debug(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise

    def process_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load and process document into chunks"""
        try:
            documents = self.load_document(file_path)
            chunks = self.split_documents(documents)
            logger.info(f"Processed {file_path}: {len(documents)} docs -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise

    def process_multiple_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """Process multiple documents"""
        all_chunks = []
        processed_count = 0
        failed_count = 0

        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                processed_count += 1
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                failed_count += 1

        logger.info(f"Processed {processed_count} documents, {failed_count} failed, {len(all_chunks)} total chunks")
        return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=== DocumentProcessor 테스트 ===")

    # 프로세서 생성
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

    # 파일 지원 확인
    print("✓ PDF 지원:", processor.is_supported_file("test.pdf"))
    print("✓ TXT 지원:", processor.is_supported_file("test.txt"))
    print("✗ 미지원:", processor.is_supported_file("test.xyz"))

    # 실제 문서가 있다면 처리 테스트
    try:
        # 현재 디렉토리의 텍스트 파일 찾기
        import os

        text_files = [f for f in os.listdir('.') if f.endswith('.txt')]

        if text_files:
            print(f"테스트 파일: {text_files[0]}")
            chunks = processor.process_document(text_files[0])
            print(f"생성된 청크: {len(chunks)}개")
            if chunks:
                print(f"첫 번째 청크: {chunks[0].page_content[:100]}...")
        else:
            print("테스트할 .txt 파일이 없습니다.")
            # 간단한 테스트 파일 생성
            test_content = "이것은 테스트 파일입니다.\n한글과 영어가 섞여있습니다.\nThis is a test file with mixed languages."
            with open('test_sample.txt', 'w', encoding='utf-8') as f:
                f.write(test_content)
            print("테스트 파일 'test_sample.txt'를 생성했습니다.")

            chunks = processor.process_document('test_sample.txt')
            print(f"생성된 청크: {len(chunks)}개")
            if chunks:
                print(f"첫 번째 청크: {chunks[0].page_content}")

    except Exception as e:
        print(f"오류: {e}")
        import traceback

        traceback.print_exc()

    print("\n테스트 완료!")
