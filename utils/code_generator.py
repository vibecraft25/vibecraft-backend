import uuid
import base64


class CodeGenerator:
    @staticmethod
    def generate_code(filename: str) -> str:
        unique_string = f"{filename}_{uuid.uuid4().hex}"
        encoded = base64.b32encode(unique_string.encode("utf-8")).decode("utf-8")
        trimmed_encoded = encoded[:32]
        return f"{trimmed_encoded}"

    @staticmethod
    def generate_code_with_ext(file: str) -> str:
        filename = file.split(".")[0]
        extension = file.split(".")[-1]
        unique_string = f"{filename}_{uuid.uuid4().hex}"
        encoded = base64.b32encode(unique_string.encode("utf-8")).decode("utf-8")
        trimmed_encoded = encoded[:32]
        return f"{trimmed_encoded}.{extension}"
