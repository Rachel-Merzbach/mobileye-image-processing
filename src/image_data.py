from PIL import Image
from pathlib import Path
import glob

ROOT = ""
IMAGE_DIR = "leftImg8bit/train"
LABEL_DIR = "gtFine/train"


class ImagesData:
    @staticmethod
    def __get_image(image):
        path = Path(f"{ROOT}/{IMAGE_DIR}/{image}_leftImg8bit.png")
        return Image.open(path)

    @staticmethod
    def __get_label(image):
        path = Path(f"{ROOT}/{LABEL_DIR}/{image}_gtFine_labelIds.png")
        return Image.open(path)

    @staticmethod
    def images_and_labeled(city):
        path = f"{ROOT}/{IMAGE_DIR}/{city}/*.png"

        for filename in glob.glob(path):
            im_path = f"{city}/{city}_{filename[-29:-16]}"

            yield ImagesData.__get_image(im_path), ImagesData.__get_label(im_path)