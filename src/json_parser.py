import json
import pathlib
import pandas as pd
import cv2
import re
import torch

from collections import defaultdict
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor


from consts import (ANN_PATH, TRAIN_IMGS_PATH, VAL_IMGS_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, TOKENIZER)

"""
    annotations.json keys:
     'info' - description of dataset
     'licenses' - information about image sources
     'images' - information about images
     'annotations' - dictionary about annotations 

annotations contain captions and images connected with these 
captions throw key 'image_id' in annotations and key 'id' in images.
"""


def yield_token(captions):
    for example in captions:
        tokens = TOKENIZER(example)
        yield tokens



class COCOparser:

    def __init__(self, json_annotations_path, img_path='train', img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        self.json_path = pathlib.Path(json_annotations_path)
        self.image_path = pathlib.Path(TRAIN_IMGS_PATH if img_path =='train' else VAL_IMGS_PATH)
        self.image_width = img_width
        self.image_height = img_height

        data = json.loads(self.json_path.read_bytes())
        # extract only filenames and id
        self.image_data = pd.DataFrame([dict(id=el['id'],
                                filename=el['file_name']) for el in data['images']])
        # extract only image_id and caption
        self.annotations_data = pd.DataFrame([dict(id=el['image_id'],
                                caption=el['caption']) for el in data['annotations']])
        
    def __get_img_name_by_id__(self, image_id):
        try:
            image_name = self.image_data.loc[self.image_data.id == image_id].filename.values[0]
        except:
            return
        
        return image_name
    
    def get_img_by_name(self, name, resize=False):
        image_path = pathlib.Path(self.image_path.as_posix() + '/' + name)
        img = cv2.imread(image_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return img
    
    def get_img_by_id(self, image_id):
        image_name = self.__get_img_name_by_id__(image_id=image_id)
        if image_name is None:
            return
        else:
            return self.get_img_by_name(image_name)
        
    def get_caption_by_id(self, image_id):
        captions = self.annotations_data.loc[self.annotations_data.id == image_id].caption.values.tolist()
        return captions if len(captions) > 0 else None
    
    def tokenize_captions(self):
        captions = self.annotations_data.caption
        captions = captions.apply(lambda x: re.sub(r"[\.,:!?]",' ', x))
        captions = captions.apply(lambda x: re.sub(r" +",' ', x))
        captions = captions.apply(lambda x: ("<|startoftext|> " + x + "<|endoftext|>").lower())
        
        #"""
        token_generator = yield_token(captions)
        vocab = build_vocab_from_iterator(token_generator)
        vocab.set_default_index(-1)
        print(vocab['<|startoftext|>'], vocab['<|endoftext|>'], vocab['bicycle'], vocab['A'])
        captions = captions.apply(lambda x: [vocab[word] for word in x.split(' ')]).values
        print(captions)
        #"""

        captions = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in captions], batch_first=True)
        print(captions.shape)
        




def display_image(img, captions):
    for caption in captions:
        cv2.imshow(caption, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

json_val_name = ANN_PATH + '/captions_val2017.json'
json_train_name = ANN_PATH + '/captions_train2017.json'
parser = COCOparser(json_train_name, img_path='train')
# parser = COCOparser(json_train_name)
# img, captions = parser.get_img_by_id(139), parser.get_caption_by_id(139)
# parser.get_dataset()
parser.tokenize_captions()