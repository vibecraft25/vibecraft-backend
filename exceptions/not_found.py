from exceptions import BaseCustomException


class NotFoundException(BaseCustomException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=404, detail=detail)
