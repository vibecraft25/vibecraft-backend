from io import BytesIO
from os import makedirs
from PIL import Image, ImageOps

from core import logger


class ImageUtils:
    @staticmethod
    async def save_image(path, file, fileName):
        if file is not None:
            try:
                makedirs(path, exist_ok=True)

                data_io = BytesIO(file)
                img = Image.open(data_io)
                img = ImageOps.exif_transpose(img)  # to prevent auto_rotate of picture
                img.save(f"{path}/{fileName}")
                # img.show()

                logger.info(f"save_image => {path}/{fileName}")

            except Exception as e:
                logger.error(e)
