import numpy as np
import numpy.typing as npt
import pathlib
import cv2
from dataclasses import dataclass
from typing import Self

@dataclass
class Mask:
    data: npt.NDArray[np.uint8] #planar RGB format
    path: pathlib.Path

def __init__(self: Mask, path: pathlib.Path) -> Mask:
    self.path = path
    array = cv2.imread(path)
    array = np.array(array, dtype = 'uint8')
    self.data = array

@classmethod
def from_path(cls, path: pathlib.Path) -> Self:
    array = cv2.imread(path)
    array = np.array(array, dtype = 'uint8')
    return cls(data = array, path = path)

@property
def getPath(self: Mask) -> pathlib.Path:
    return self.path

@property
def nShips(self: Mask) -> int:
    contours, _ = cv2.findContours(self.data, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)