"""
Processes the MASATI v2 dataset to create its masks.

This script uses the internal utilities of the shipseg module to create
masks for each of the MASATI v2 images. The created masks are stored in
TIFF format.
"""

from pathlib import Path

from tqdm import tqdm

from shipseg.data.preprocess.annotation_to_mask import Annotation2Mask
from shipseg.utils.config import DATASET_PATH



if __name__ == '__main__':

    # Step 1: Setting up the mask creation process
    dataset_path = Path(DATASET_PATH)
    image_folder = dataset_path / 'images'
    annotations_folder = dataset_path / 'labels'
    (mask_folder := dataset_path / 'masks').mkdir(exist_ok=True, parents=True)

    # Step 2: Creating masks by iterating the dataset folder
    for img_file in tqdm(list(image_folder.glob('*.png')),
                         desc='Creating masks', unit='mask'):
        try:
            annotation_file = annotations_folder / f'{img_file.stem}.txt'
            annotation2mask = Annotation2Mask.from_file(
                img_file=img_file,
                annotation_file=annotation_file
            )
            annotation2mask.create_mask().tofile(
                mask_folder / f'{img_file.stem}.bin'
            )
        except FileNotFoundError:
            print('[MASKGenerator] Could not create the mask for '
                  f'"{img_file.name}" due to missing files')
