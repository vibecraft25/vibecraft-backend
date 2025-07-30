from exceptions import BaseCustomException


class UnauthorizedException(BaseCustomException):
    def __init__(self, detail: str = "NOT AUTHORIZED"):
        super().__init__(status_code=401, detail=detail)
