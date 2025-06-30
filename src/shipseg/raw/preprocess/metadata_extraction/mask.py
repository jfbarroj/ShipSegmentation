from typing import Self, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

from shipseg.raw.preprocess.metadata_extraction.image import Image
from shipseg.utils.config import DATASET_PATH
@dataclass
class Mask:
    def __init__(self, data: npt.NDArray[np.uint8], path: Path):
        self.data = data
        self.path = path

    @classmethod
    def from_path(cls, path: Path) -> Self:
        img_path = Path(DATASET_PATH) / 'images' / f'{path.stem}.png'
        image = Image.from_path(img_path=img_path)
        array = np.fromfile(path, dtype='uint8')
        array = array.reshape((image.height,image.width))
        return cls(data=array, path=path)

    @property
    def blobs(self) -> Tuple[np.ndarray]:
        contours, _ = cv2.findContours(self.data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @property
    def n_blobs(self) -> int:
        return len(self.blobs)
