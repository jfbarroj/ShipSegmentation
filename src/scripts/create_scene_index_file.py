from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from shipseg.utils.config import DATASET_PATH, PROCESSED_DATASET_PATH
from shipseg.raw.preprocess.metadata_extraction.image_metadata import ImageMetadata


if __name__ == '__main__':
    
    # Step 1: Defining the dataset paths
    dataset_path = Path(DATASET_PATH)
    (processed_dataset_path := Path(PROCESSED_DATASET_PATH)).mkdir(exist_ok=True, parents=True)
    images_directory = dataset_path / 'images'
    masks_directory = dataset_path / 'masks'

    # Step 2: Calculating MASATI scene metadata
    image_metadata_list: List[Dict[str, Any]] = []

    try:
        for image in tqdm(list(images_directory.glob('*.png')),
                          desc='Extracting metadata from images',
                          unit='image'):

            # Step 2.1: Load scene metadata
            metadata = ImageMetadata.from_paths(image, masks_directory / f'{image.stem}.bin')

            # Step 2.2: Temporarily store scene metadata
            image_metadata_list.append(vars(metadata))

        # Step 2.3: Storing MASATI metadata
        pd.DataFrame(image_metadata_list).to_csv(processed_dataset_path / 'index.csv', index=False)
    except(FileNotFoundError):
        print('A non existing filename / directory  name was given')
    except IOError:
        print('An error ocurred while writing into metadata,csv file')
    except ValueError:
        print('A wrong parameter has been given')
    except Exception:
        print('An error ocurred')