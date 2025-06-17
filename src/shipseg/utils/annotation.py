"""
Implements an abstraction for MASATI v2 annotations.

This module implements a class to offer an abstraction for a bounding box
based on its location (`BoundingBox`), an abstraction to manage bounding
boxes based on their center (`CenteredBoundingBox`), and another abstraction
to manage the sequence of bounding boxes described in an annotation
of the MASATI v2 dataset.
"""

from abc import abstractmethod
from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import List, Self


@dataclass
class IBoundingBox:
    """
    Implements an interface for all bounding boxes.

    Attributes:
        class_id (int): the class identifier of the bbox.

    Properties:
        is_relative (bool): determines if the bbox is relative.
        is_absolute (bool): determines if the bbox is absolute.
    """

    class_id: int


    @property
    @abstractmethod
    def is_relative(self) -> bool:
        """Determines if the bbox is relative."""


    @property
    def is_absolute(self) -> bool:
        """Determines if the bbox is absolute."""
        return not self.is_relative


    @abstractmethod
    def to_relative(self, img_width: int, img_height: int) -> Self:
        """
        Creates a new object of `Self` type with relative coords.

        Args:
            img_width (int): the absolute width of the whole image.
            img_height (int): the absolute height of the whole image.

        Returns:
            A `Self`-typed object with relative coordinates.
        """


    @abstractmethod
    def to_absolute(self, img_width: int, img_height: int) -> Self:
        """
        Creates a new object of `Self` type with absolute coords.

        Args:
            img_width (int): the absolute width of the whole image.
            img_height (int): the absolute height of the whole image.

        Returns:
            A `Self`-typed object with absolute coordinates.
        """


@dataclass
class BoundingBox(IBoundingBox):
    """
    Implements an abstraction for the location of a bounding box.

    Attributes:
        class_id (int | float): the identifier of the class of the bbox.
        min_x (int | float): the coord of the first X-axis pixel of the bbox.
        max_x (int | float): the coord of the last X-axis pixel of the bbox.
        min_y (int | float): the coord of the first Y-axis pixel of the bbox.
        max_y (int | float): the coord of the last Y-axis pixel of the bbox.

    Properties:
        is_relative (bool): determines if the bbox is relative.
        is_absolute (bool): determines if the bbox is absolute.
    """


    min_x: int | float
    max_x: int | float
    min_y: int | float
    max_y: int | float


    @property
    def is_relative(self) -> bool:
        check_is_relative = lambda coordinate: 0 <= coordinate <= 1

        return check_is_relative(self.min_x) and\
               check_is_relative(self.max_x) and\
               check_is_relative(self.min_y) and\
               check_is_relative(self.max_y)


    def to_absolute(self, img_width: int, img_height: int) -> Self:
        """
        Creates a BoundingBox object from self with absolute coords.

        Args:
            img_width (int): the absolute width of the whole image.
            img_height (int): the absolute height of the whole image.

        Returns:
            A BoundingBox object with absolute coordinates.
        """
        if not self.is_absolute:
            return BoundingBox(
                class_id=self.class_id,
                min_x=int(self.min_x * img_width),
                max_x=int(self.max_x * img_width),
                min_y=int(self.min_y * img_height),
                max_y=int(self.max_y * img_height)
            )

        return BoundingBox(
            class_id=self.class_id,
            min_x=self.min_x,
            max_x=self.max_x,
            min_y=self.min_y,
            max_y=self.max_y
        )


    def to_relative(self, img_width: int, img_height: int) -> Self:
        """
        Creates a BoundingBox object from self with relative coords.

        Args:
            img_width (int): the absolute width of the whole image.
            img_height (int): the absolute height of the whole image.

        Returns:
            A BoundingBox object with relative coordinates.
        """
        if not self.is_relative:
            return BoundingBox(
                class_id=self.class_id,
                min_x=self.min_x / img_width,
                max_x=self.max_x / img_width,
                min_y=self.min_y / img_height,
                max_y=self.max_y / img_height
            )

        return BoundingBox(
            class_id=self.class_id,
            min_x=self.min_x,
            max_x=self.max_x,
            min_y=self.min_y,
            max_y=self.max_y
        )


    def to_centered_bbox(self) -> 'CenteredBoundingBox':
        """Converts the current bbox to a center-based bbox object."""
        return CenteredBoundingBox(
            class_id=self.class_id,
            center_x=(self.min_x + self.max_x) / 2,
            center_y=(self.min_y + self.max_y) / 2,
            width=self.max_x - self.min_x,
            height=self.max_y - self.min_y,
        )


@dataclass
class CenteredBoundingBox(IBoundingBox):
    """
    Implements an abstraction for the annotation of a single bounding box.

    Attributes:
        class_id (int): the identifier of the detected class.
        center_x (int | float): the absolute/relative X-axis coordinate of the
            bbox center.
        center_y (int | float): the absolute/relative Y-axis coordinate of the
            bbox center.
        width (int | float): the absolute/relative width of the bbox.
        height (int | float): the absolute/relative height of the bbox.

    Properties:
        is_relative (bool): determines if the bbox is relative.
        is_absolute (bool): determines if the bbox is absolute.
    """


    center_x: int | float
    center_y: int | float
    width: int | float
    height: int | float


    @property
    def is_relative(self) -> bool:
        check_is_relative = lambda coordinate: 0 <= coordinate <= 1

        return check_is_relative(self.center_x) and\
               check_is_relative(self.center_y) and\
               check_is_relative(self.width) and\
               check_is_relative(self.height)


    def to_absolute(self, img_width: int, img_height: int) -> Self:
        """
        Creates a CenteredBoundingBox object from self with absolute coords.

        Args:
            img_width (int): the absolute width of the whole image.
            img_height (int): the absolute height of the whole image.

        Returns:
            A CenteredBoundingBox object with absolute coordinates.
        """
        if not self.is_absolute:
            return CenteredBoundingBox(
                class_id=self.class_id,
                center_x=int(self.center_x * img_width),
                center_y=int(self.center_y * img_height),
                width=int(self.width * img_width),
                height=int(self.height * img_height)
            )

        return CenteredBoundingBox(
            class_id=self.class_id,
            center_x=self.center_x,
            center_y=self.center_y,
            width=self.width,
            height=self.height
        )


    def to_relative(self, img_width: int, img_height: int) -> Self:
        """
        Creates a CenteredBoundingBox object from self with relative coords.

        Args:
            img_width (int): the absolute width of the whole image.
            img_height (int): the absolute height of the whole image.

        Returns:
            A CenteredBoundingBox object with relative coordinates.
        """
        if not self.is_relative:
            return CenteredBoundingBox(
                class_id=self.class_id,
                center_x=self.center_x / img_width,
                center_y=self.center_y / img_height,
                width=self.width / img_width,
                height=self.height / img_height
            )

        return CenteredBoundingBox(
            class_id=self.class_id,
            center_x=self.center_x,
            center_y=self.center_y,
            width=self.width,
            height=self.height
        )


    def to_bbox(self) -> BoundingBox:
        """Converts the current center-based bbox to a normal bbox object."""
        return BoundingBox(
            class_id=self.class_id,
            min_x=self.center_x - self.width / 2,
            max_x=self.center_x + self.width / 2,
            min_y=self.center_y - self.height / 2,
            max_y=self.center_y + self.height / 2,
        )


    @classmethod
    def from_str(cls, centered_bbox_str: str) -> Self:
        """
        Creates a CenteredBoundingBox object from a string representation.

        NOTE: this method returns `None` an empty string is given. Empty
        strings represent no detections.

        Args:
            centered_bbox_str: the string representation of the bbox.

        Returns:
            A CenteredBoundingBox object with the information provided by
            the string representation.

        Raises:
            ValueError: if the given string is not empty nor a valid
                representation of the centered bbox.
        """

        # Step 1: Obtaining the bbox information (if any)
        bbox_data = centered_bbox_str.split()

        # Step 2: Checking for empty strings and wrong representations
        if len(bbox_data) == 0:
            return None

        if len(bbox_data) != 5:
            return

        # Step 3: Returning the centered bounding box object
        return cls(class_id=int(bbox_data[0]),
                   center_x=float(bbox_data[1]),
                   center_y=float(bbox_data[2]),
                   width=float(bbox_data[3]),
                   height=float(bbox_data[4]))


@dataclass
class MASATIImageAnnotation:
    """
    Implements an abstraction for MASATI v2 image annotations.

    Attributes:
        _raw_annotations (List[CenteredBoundingBox]): the "raw" annotations of
            an image. They must be normalized CenteredBoundingBox objects.

    Properties:
        centered_annotations (List[CenteredBoundingBox]): provides the
            annotations of the image as centered annotations.
        annotations (List[BoundingBox]): provides the annotations of the image
            as bboxes defined by the four points of the bbox shape.
    """


    _raw_annotations: InitVar[List[CenteredBoundingBox] | None] = []


    def __post_init__(
            self,
            _raw_annotations: List[CenteredBoundingBox] | None
        ) -> None:
        """
        Provides a default empty list value for raw annotations when omitted.

        Args:
            _raw_annotations (List[CenteredBoundingBox] | None): the raw
                annotations of the MASATIImageAnnotation object to be created.
        """
        self._raw_annotations = [] if _raw_annotations is None\
                                else _raw_annotations


    @property
    def centered_annotations(self) -> List[CenteredBoundingBox]:
        """Retrieves the CenteredBoundingBox objects of the image"""
        return self._raw_annotations


    @property
    def annotations(self) -> List[BoundingBox]:
        """Retrieves the BoundingBox objects of the image"""
        return [centered_bbox.to_bbox()
                for centered_bbox in self._raw_annotations]


    @classmethod
    def from_str(cls, image_annotations: str | List[str]) -> Self:
        """
        Creates a MASATIImageAnnotation object from a string representation.

        Args:
            image_annotations_str (str | List[str]): a string with one or
                more bbox representations separated by a new line character,
                or a list of string representations of each bbox.

        Returns:
            A MASATIImageAnnotation object with the provided annotations.
        """

        # Step 1: Checking for str vs List[str]
        if isinstance(image_annotations, str):
            image_annotations: List[str] = image_annotations.split('\n')

        # Step 2: Creating the annotation object
        return cls(
            _raw_annotations=[
                    CenteredBoundingBox.from_str(bbox_repr)
                    for bbox_repr in image_annotations
                    if bbox_repr != ''
            ]
        )


    @classmethod
    def from_file(cls, annotation_file: str | Path) -> Self:
        """
        Creates a MASATIImageAnnotation object from an annotation file.

        Args:
            annotation_file (str | Path): the path to the annotation file.

        Returns:
            A MASATIImageAnnotation object with the provided annotations.

        Raises:
            FileNotFoundError: if the file could not be found.
        """

        # Step 1: Converting the annotation file to a path object
        annotation_file: Path = Path(annotation_file)

        # Step 2: Checking for file existence
        if not annotation_file.exists() or not annotation_file.is_file():
            raise FileNotFoundError(f'Could not find file "{annotation_file}"')

        # Step 3: Reading the file
        with open(annotation_file, encoding='utf-8', mode='r') as ann_fd:
            annotation_data = ann_fd.read()

        # Step 4: Creating the annotation object
        return cls.from_str(image_annotations=annotation_data)
