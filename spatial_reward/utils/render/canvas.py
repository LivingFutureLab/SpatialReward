import typing as t

from PIL import Image, ImageColor, ImageOps, ImageChops, ImageFilter
import numpy as np


class Canvas:
    def __init__(self, filename=None, height=500, width=500):
        self.filename = filename
        self.height, self.width = height, width
        self.img = Image.new("RGBA", (self.height, self.width), (0, 0, 0, 0))

    def draw(self, dots, color: t.Union[tuple, str]):
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        if isinstance(dots, tuple):
            dots = [dots]
        for dot in dots:
            if dot[0] >= self.height or dot[1] >= self.width or dot[0] < 0 or dot[1] < 0:
                continue
            self.img.putpixel(dot, color + (255,))

    def add_white_border(self, border_size=5):
        if self.img.mode != "RGBA":
            self.img = self.img.convert("RGBA")
        alpha = self.img.getchannel("A")
        dilated_alpha = alpha.filter(ImageFilter.MaxFilter(size=5))
        white_area = Image.new("RGBA", self.img.size, (255, 255, 255, 255))
        white_area.putalpha(dilated_alpha)
        result = Image.alpha_composite(white_area, self.img)
        return result

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.filename:
            self.img.save(self.filename)
