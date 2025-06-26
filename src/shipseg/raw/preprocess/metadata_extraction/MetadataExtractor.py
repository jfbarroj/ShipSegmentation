import numpy as np
import numpy.typing as npt
import pathlib
import cv2
from dataclasses import dataclass
from typing import Self
from shipseg.raw.preprocess.metadata_extraction.Image import Image
from shipseg.raw.preprocess.metadata_extraction.Mask import Mask

@dataclass
class MetadataExtractor:
    img_id: str
    img_path: pathlib.Path
    mask_path: pathlib.Path
    nbands: int
    min_red: int
    min_green: int
    min_blue: int
    max_red: int
    max_green: int
    max_blue: int
    nShips: int

    @classmethod
    def from_paths(cls,img_path: pathlib.Path, mask_path: pathlib.Path) -> Self:
        image = Image.from_path(img_path)
        mask = Mask.from_path(mask_path)
        return cls(img_id = image.getImgId, img_path = img_path, mask_path = mask_path,
                   nbands = image.getNbands, min_red = image.getMin(0), min_green = image.getMin(1),
                   min_blue = image.getMin(2), max_red = image.getMax(0), max_green = image.getMax(1),
                   max_blue = image.getMax(2), nships = mask.nShips)
        