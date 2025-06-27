from typing import Self, Tuple

import numpy as np
import numpy.typing as npt
import pathlib
import cv2
from dataclasses import dataclass

@dataclass
class Mask:
    def __init__(self, data: npt.NDArray[np.uint8], path: pathlib.Path):
        self.data = data
        self.path = path

    @classmethod
    def from_path(cls, path: pathlib.Path) -> Self:
        mask_data = cv2.imread(path)
        return cls(data = mask_data, path = path)

    @property
    def blobs(self) -> Tuple[np.ndarray]:
        contours, _ = cv2.findContours(self.data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @property
    def n_blobs(self) -> int:
        return len(self.blobs)
