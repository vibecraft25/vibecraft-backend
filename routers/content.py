__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio
import json

# Third-party imports
from fastapi import APIRouter
from fastapi import UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# Custom imports
from schemas.data_schemas import DatasetMetadata
from utils import PathUtils, CodeGenerator, ImageUtils
from exceptions import NotFoundException

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


@router.get("/meta", response_model=DatasetMetadata)
async def get_meta(
    thread_id: str,
):
    if PathUtils.is_exist(thread_id, f"{thread_id}_meta.json"):
        file_path = PathUtils.get_path(thread_id, f"{thread_id}_meta.json")
        with open(file_path[0], 'r', encoding='utf-8') as f:
            meta_content = f.read()
            meta_data = json.loads(meta_content)
        return JSONResponse(content=jsonable_encoder(DatasetMetadata(**meta_data)))
    raise NotFoundException(detail=f"Meta Resource Not Found: {thread_id}")
