"""
Transforms a YOLO-based annotation of a MASATI v2 scene into a TIFF mask.

This module implements a mask generation functionality that converts the
annotations provided within the MASATI v2 dataset into binary masks that
can be used for training ship segmentation models.

Note: the YOLO-based annotation expected follows the following format:

`(0, normalized_center_x, normalized_center_y, width, height)`
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np

from shipseg.utils.annotation import MASATIImageAnnotation

@dataclass
class Annotation2Mask:
    """
    Implements the transformormation of a YOLO annotation to a mask.

    Attributes:
        img (np.ndarray): the image whose mask will be obtained.
        annotations (MASATIImageAnnotation): the annotation of the image.
    """


    img: np.ndarray
    annotations: MASATIImageAnnotation


    def __post_init__(self) -> None:
        """
        Asserts that the created object is correct.
        """

        # Step 1: Check that the image is three-dimensional
        if self.img.ndim > 3:
            raise ValueError('Found image with more than three dimensions '
                             f'(dims={self.img.ndim})')

        # Step 2: Check that the image has exactly 3 channels
        if self.img.shape[0] != 3 and self.img.shape[-1] != 3:
            raise ValueError('Found image with more than three channels or '
                             'unknown format')

        # Step 3: If interleaved, convert to planar
        if self.img.shape[-1] == 3:
            self.img = self.img.transpose(2, 0, 1)


    def create_mask(self) -> np.ndarray:
        """
        Creates a mask from the image and its annotation.

        Returns:
            A NumPy array containing the binary mask of the image.
        """

        # Step 1: Create zeros mask
        mask = np.zeros(self.img.shape[1:], dtype=np.uint8)

        # Step 2: Draw bboxes in the mask
        for relative_bbox in self.annotations.annotations:
            bbox = relative_bbox.to_absolute(img_width=self.img.shape[2],
                                             img_height=self.img.shape[1])
            mask[bbox.min_y : bbox.max_y, bbox.min_x : bbox.max_x] = 255

        return mask


    @classmethod
    def from_file(cls, img_file: str | Path,
                  annotation_file: str | Path) -> Self:
        """
        Creates a new Annotation2Mask object from img and annotation files.

        Args:
            img_file (str | Path): the path to the image.
            annotation_file (str | Path): the path of the annotation file.
        """
        def assert_file_exists(file: Path) -> None:
            """Asserts the given file exists and is a file"""
            if not file.exists() or not file.is_file():
                raise FileNotFoundError(f'Could not find the file: "{file}"')

        # Step 1: Checking the given files
        img_file: Path = Path(img_file)
        annotation_file: Path = Path(annotation_file)

        assert_file_exists(img_file)
        assert_file_exists(annotation_file)

        # Step 2: Creating the Annotation2Mask object
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)[...,:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        annotations = MASATIImageAnnotation.from_file(annotation_file)

        return cls(img=img, annotations=annotations)
