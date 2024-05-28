#def segment_image():

import tifffile
import os
import json
import pandas as pd

image_path = 'data/sample_data/roof/rawdata/test/prediction image_1.1.tif'
saved_dir = 'data/sample_data/roof/rawdata/test'

def crop_image(image_path, saved_dir, croped_dim=(1280, 1280), overlap_percentage=0):
    """Crops an image into smaller, non-overlapping or overlapping tiles.

        Args:
            image_path (str): Path to the image file.
            saved_dir (str): Directory path to save the cropped tiles.
            cropped_dim (tuple, optional): Desired dimensions for each cropped tile. Defaults to (1280, 1280).
            overlap_percentage (float, optional): Percentage of overlap between adjacent tiles (0 for non-overlapping). Defaults to 0.
                    Values between 0 and 100 are allowed.
        
        Raises:
            ValueError: If the overlap_percentage is outside the valid range (0 to 100).
            FileNotFoundError: If the image file is not found.
            OSError: If there's an error creating the saved directory.

    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not 0 <= overlap_percentage <= 100:
        raise ValueError("overlap_percentage must be between 0 and 100")

    # Create the saved directory if it doesn't exist
    os.makedirs(saved_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    saved_prefix = os.path.join(saved_dir, 'croped_'+ image_name + '_')


    df_coordinates = pd.DataFrame(columns={'left':None, 'top':None, 'right':None, 'bottom':None})
    
    # Crop image
    with tifffile.TiffFile(image_path) as tif:
        # Convert to array
        image = tif.asarray()

        # Get image dimension
        img_height, img_width = image.shape[:2]

        crop_width = croped_dim[0]
        crop_height = croped_dim[1]

        # Calculate step size
        step_x = int(crop_width * (1 - (overlap_percentage/100)))
        step_y = int(crop_height * (1 - (overlap_percentage/100)))

        crops = []

        for top in range(0, img_height - crop_height + step_y, step_y):
            for left in range(0, img_width -crop_width + step_x, step_x):
                # Define bottom and right coordinates
                bottom = top + crop_height
                right = left + crop_width

                top_adjusted = max(0, img_height - crop_height) if bottom > img_height else top
                left_adjusted = max(0, img_width - crop_width) if right > img_width else left

                bottom_adjusted = min(bottom, img_height)
                right_adjusted = min(right, img_width)

                crop = image[top_adjusted:bottom_adjusted, left_adjusted:right_adjusted]
                
                loc = 'top' + str(top) + 'left' + str(left)

                # write the cropped image
                saved_path = saved_prefix + loc + '.tif'
                tifffile.imwrite(saved_path, crop)
                
                # write coordinate
                df_coordinates.loc[loc] = {'left':left_adjusted, 'top':top_adjusted, 'right':right_adjusted, 'bottom':bottom_adjusted}


    # save coordinate inf. 
    saved_coordinates_file = saved_prefix + 'coordinates.xlsx'   
    df_coordinates.to_csv(saved_coordinates_file, index=True) 

    print("croped images and its corresponding coordinates in " + saved_dir)
    
    return 