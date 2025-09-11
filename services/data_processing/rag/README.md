# RAG (Retrieval-Augmented Generation) Module

이 모듈은 문서 기반 RAG 시스템의 핵심 컴포넌트들을 제공합니다.

## 구조

```
rag/
├── __init__.py              # 모듈 exports
├── chroma_db.py            # ChromaDB 벡터 저장소 래퍼
├── document_processor.py   # 문서 처리 및 청킹
└── README.md             
```

## 주요 클래스

### ChromaDB

벡터 데이터베이스를 위한 Chroma 래퍼 클래스

```python
from services.data_processing.rag import ChromaDB

# 초기화
chroma_db = ChromaDB(
    persist_directory="./chroma_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    collection_name="documents"
)

# 문서 추가
chroma_db.add_documents(documents, ids=["doc1", "doc2"])

# 유사도 검색
results = chroma_db.similarity_search("질문 텍스트", k=5)

# 점수와 함께 검색
results_with_scores = chroma_db.similarity_search_with_score("질문 텍스트", k=5)
```

**주요 메서드:**
- `add_documents(documents, ids)`: 문서 추가
- `similarity_search(query, k, filter_dict)`: 유사도 검색
- `similarity_search_with_score(query, k, filter_dict)`: 점수와 함께 유사도 검색
- `delete_documents(ids, where)`: 문서 삭제
- `get_documents(ids, where)`: 문서 조회
- `as_retriever(search_kwargs)`: Retriever 인터페이스로 변환
- `reset_collection()`: 컬렉션 초기화

### DocumentProcessor

문서 로딩과 텍스트 청킹을 담당하는 클래스

```python
from services.data_processing.rag import DocumentProcessor

# 초기화
processor = DocumentProcessor(
    chunk_size=600,
    chunk_overlap=0
)

# 단일 문서 처리
chunks = processor.process_document("path/to/document.pdf")

# 여러 문서 처리
all_chunks = processor.process_multiple_documents([
    "doc1.pdf", 
    "doc2.txt", 
    "doc3.md"
])

# 지원 파일 형식 확인
is_supported = processor.is_supported_file("document.pdf")
```

**지원 파일 형식:**
- PDF (`.pdf`)
- 텍스트 (`.txt`)
- 엑셀 (`.xlsx`, `.xls`)
- 마크다운 (`.md`, `.markdown`)

**주요 메서드:**
- `process_document(file_path)`: 단일 문서를 로드하고 청킹
- `process_multiple_documents(file_paths)`: 여러 문서 처리
- `load_document(file_path)`: 문서 로드
- `split_documents(documents)`: 문서 청킹
- `is_supported_file(file_path)`: 지원 형식 확인

## 사용 예제

### 기본 RAG 파이프라인

```python
from services.data_processing.rag import ChromaDB, DocumentProcessor

# 1. 문서 처리기 초기화
processor = DocumentProcessor(chunk_size=600, chunk_overlap=50)

# 2. 벡터 DB 초기화
chroma_db = ChromaDB(persist_directory="./vector_db")

# 3. 문서 처리 및 인덱싱
documents = processor.process_document("document.pdf")
chroma_db.add_documents(documents)

# 4. 검색
results = chroma_db.similarity_search("검색 질의", k=5)

for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### LangChain과 함께 사용

```python
# Retriever로 변환하여 체인에서 사용
retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

# LangChain 체인에서 사용
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=retriever
)
```

### 여러 문서 처리

```python
from pathlib import Path

# 여러 문서 처리
doc_paths = [
    "manual1.pdf",
    "guide.txt", 
    "notes.md"
]

processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
all_documents = processor.process_multiple_documents(doc_paths)

# 벡터 DB에 추가
chroma_db.add_documents(all_documents)

print(f"총 {len(all_documents)}개 청크가 인덱싱되었습니다.")
```

## 설정

### 환경 변수

- `CHROMA_PATH`: ChromaDB 저장 경로 (기본값: config.settings.chroma_path)

### 임베딩 모델

기본적으로 `sentence-transformers/all-MiniLM-L6-v2` 모델을 사용합니다. 다른 모델을 사용하려면:

```python
chroma_db = ChromaDB(embedding_model="sentence-transformers/all-mpnet-base-v2")
```

### 청킹 설정

```python
processor = DocumentProcessor(
    chunk_size=1000,      # 최대 청크 크기
    chunk_overlap=200,    # 청크 간 겹치는 부분
    separators=["\n\n", "\n", ".", "!", "?"]  # 분할 기준
)
```

## 로깅

모든 클래스는 Python 표준 logging을 사용합니다. 로그 레벨을 조정하여 디버깅 정보를 확인할 수 있습니다:

```python
import logging
logging.getLogger('services.data_processing.rag').setLevel(logging.DEBUG)
```

## 의존성

- `langchain-community`: 문서 로더와 벡터 저장소
- `langchain`: 핵심 LangChain 라이브러리
- `sentence-transformers`: 텍스트 임베딩 생성
- `chromadb`: 벡터 데이터베이스