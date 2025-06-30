from typing import Self
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

from shipseg.utils.config import DATASET_PATH

@dataclass
class Image:
    def __init__(self, data: npt.NDArray[np.uint8], path: Path):
        self.data = data
        self.path = path

    @classmethod
    def from_path(cls, img_path: Path) -> Self:
        array: cv2.typing.MatLike = cv2.imread(img_path.resolve())
        array = array.transpose(2, 0, 1)
        array = array[::-1]
        return cls(data = array, path = img_path.relative_to(Path(DATASET_PATH)))

    @property
    def img_id(self) -> str:
        return self.path.stem

    @property
    def n_bands(self) -> int:
        return self.data.shape[0]
    
    @property
    def height(self) -> int:
        return self.data.shape[1]
    
    @property
    def width(self) -> int:
        return self.data.shape[2]

    def get_min(self, channel: int) -> int:
        return int(self.data[channel].min())

    def get_max(self, channel: int) -> int:
        return int(self.data[channel].max())