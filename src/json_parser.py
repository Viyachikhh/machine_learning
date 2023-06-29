import json
import pathlib
import pandas as pd
import cv2

from collections import defaultdict

from consts import (ANN_PATH, TRAIN_IMGS_PATH, VAL_IMGS_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)

"""
    annotations.json keys:
     'info' - description of dataset
     'licenses' - information about image sources
     'images' - information about images
     'annotations' - dictionary about annotations 

annotations contain captions and images connected with these 
captions throw key 'image_id' in annotations and key 'id' in images.
"""


class COCOparser:

    def __init__(self, json_annotations_path, img_path='train'):
        self.json_path = pathlib.Path(json_annotations_path)
        self.image_path = pathlib.Path(TRAIN_IMGS_PATH if img_path =='train' else VAL_IMGS_PATH)

        data = json.loads(self.json_path.read_bytes())
        # extract only filenames and id
        self.image_data = pd.DataFrame([dict(id=el['id'],
                                filename=el['file_name']) for el in data['images']])
        # extract only image_id and caption
        self.annotations_data = pd.DataFrame([dict(image_id=el['image_id'],
                                caption=el['caption']) for el in data['annotations']])

    
    def get_img_and_caption_by_id(self, image_id, return_path=False):
        """
        If True - return path of image, else - return image in numpy
        """
        try:
            image_name = self.image_data.loc[self.image_data.id == image_id].filename.values[0]
        except:
            return 'No images found'
        
        captions = self.annotations_data.loc[self.annotations_data.image_id == image_id].caption.values.tolist()
        
        image_path = pathlib.Path(self.image_path.as_posix() + '/' + image_name)
        if return_path:
            return image_path.as_posix(), captions
        
        img = cv2.imread(image_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, captions


def display_image(img, captions):
    for caption in captions:
        cv2.imshow(caption, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

json_val_name = ANN_PATH + '/captions_val2017.json'
json_train_name = ANN_PATH + '/captions_train2017.json'
parser = COCOparser(json_val_name, img_path='val')
# parser = COCOparser(json_train_name)
img, captions = parser.get_img_and_caption_by_id(139)
display_image(img, captions)