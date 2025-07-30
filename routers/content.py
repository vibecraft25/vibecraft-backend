import asyncio
from fastapi import APIRouter
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

from utils import PathUtils, CodeGenerator, ImageUtils

prefix = "contents"
router = APIRouter(prefix=f"/{prefix}", responses={401: {"description": "raw data file upload"}})


@router.post("/upload", status_code=201)
async def upload(
    thread_id: str,
    file: UploadFile = File(...),
):

    contents = await file.read()
    path = PathUtils.generate_path(thread_id)
    file_name = CodeGenerator.generate_code_with_ext(file.filename)
    asyncio.create_task(ImageUtils.save_image(path, contents, file_name))

    return JSONResponse(content={"code": file_name}, status_code=201)
