from dataclasses import dataclass
from pathlib import Path
from typing import Self

from shipseg.raw.preprocess.metadata_extraction.image import Image
from shipseg.raw.preprocess.metadata_extraction.mask import Mask

@dataclass
class ImageMetadata:

    img_id: str
    img_path: Path
    mask_path: Path
    n_bands: int
    height: int
    width: int
    n_ships: int
    min_red: int
    min_green: int
    min_blue: int
    max_red: int
    max_green: int
    max_blue: int
        

    @classmethod
    def from_paths(cls, img_path: Path, mask_path: Path) -> Self:
        # TODO: USE RELATIVE_TO (MASATI as main folder)
        image = Image.from_path(img_path)
        mask = Mask.from_path(mask_path)
        return cls(img_id=image.img_id, img_path=img_path, mask_path=mask_path,
                   n_bands=image.n_bands, height=image.height, width=image.width, min_red=image.get_min(0), min_green=image.get_min(1),
                   min_blue=image.get_min(2), max_red=image.get_max(0), max_green=image.get_max(1),
                   max_blue=image.get_max(2), n_ships=mask.n_blobs)
