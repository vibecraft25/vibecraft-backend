from glob import glob
from datetime import datetime

from exceptions import NotFoundException
from config import settings


class PathUtils:

    @staticmethod
    def generate_path(user: str) -> str:
        return f"{settings.file_path}/{user}/{datetime.now().strftime('%y%m')}"

    @staticmethod
    def get_path(user: str, file_name: str) -> list[str]:
        return glob(f"{settings.file_path}/{user}/*/**/{file_name}", recursive=True)

    @staticmethod
    def is_exist(user: str, file_name: str) -> bool:
        paths = glob(f"{settings.file_path}/{user}/*/**/{file_name}", recursive=True)
        if len(paths) == 0:
            raise NotFoundException(detail=f"Resource Not Found: {file_name}")
        return True
