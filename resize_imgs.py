import glob
from PIL import Image, ImageOps
import os


class ImageProcessor:

    def __init__(self, img_size):
        self.img_size = img_size

    def resize_imgs(self, path):
        for i, img_path in enumerate(glob.glob(path)):
            print(i, img_path)
            img = Image.open(img_path)
            img = ImageOps.fit(img, (self.img_size[0], self.img_size[1]), Image.ANTIALIAS)
            name = img_path.split("\\")[-1].split('.')[0]
            if not os.path.exists(f"resized_{self.img_size[0]}_{self.img_size[1]}"):
                os.makedirs(f"resized_{self.img_size[0]}_{self.img_size[1]}")
            img.save(f"resized_{self.img_size[0]}_{self.img_size[1]}/{name}.png")


imgprocessor = ImageProcessor((256, 192))
imgprocessor.resize_imgs("data")
