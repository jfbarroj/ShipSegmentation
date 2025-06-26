import numpy as np
import numpy.typing as npt
import pathlib
import cv2
from dataclasses import dataclass
from typing import Self

@dataclass
class Image:
    data: npt.NDArray[np.uint8] #RGB planar format
    path: pathlib.Path

    def __init__(self, path: pathlib.Path):
        self.path = path
        array = cv2.imread(path)
        array = np.array(array, dtype = 'uint8')
        array = array.transpose(2,0,1)
        array = array[::-1]
        self.data = array

    @classmethod
    def from_path(cls, img_path: pathlib.Path) -> Self:
        array = cv2.imread(img_path)
        array = np.array(array, dtype = 'uint8')
        array = array.transpose(2,0,1)
        array = array[::-1]
        return cls(data = array, path = img_path)

    @property
    def getPath(self) -> pathlib.Path:
        return self.path

    @property
    def getImgId(self) -> str:
        return self.path.stem

    @property
    def getNbands(self) -> int:
        return self.data.shape[0]

    def getMin(self, channel: int) -> np.uint8:
        return self.data[channel].min()

    def getMax(self, channel: int) -> np.uint8:
        return self.data[channel].max()