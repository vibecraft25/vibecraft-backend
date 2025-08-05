__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
from os import makedirs

# Custom imports
from core import logger


class ContentUtils:
    ALLOWED_EXTENSIONS = {'.csv', '.sqlite', '.sqlite3', '.db'}

    @staticmethod
    async def save_file(path: str, file: bytes, file_name: str):
        if file is not None:
            try:
                # 파일 확장자 검증
                file_extension = os.path.splitext(file_name)[1].lower()
                if file_extension not in ContentUtils.ALLOWED_EXTENSIONS:
                    raise ValueError(f"허용되지 않은 파일 형식입니다. 허용 형식: {', '.join(ContentUtils.ALLOWED_EXTENSIONS)}")

                makedirs(path, exist_ok=True)

                # 파일 저장
                file_path = f"{path}/{file_name}"
                with open(file_path, 'wb') as f:
                    f.write(file)

                logger.info(f"save_file => {file_path}")

            except Exception as e:
                logger.error(f"파일 저장 중 오류 발생: {e}")
                raise
