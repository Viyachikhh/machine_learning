
import json
import pathlib
import polars as pl
import cv2
import re
import torch
import time


from collections import defaultdict
from gc import collect
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor
from pandarallel import pandarallel


from consts import (ANN_PATH, TRAIN_IMGS_PATH, VAL_IMGS_PATH, 
                    IMAGE_WIDTH, IMAGE_HEIGHT, 
                    TOKENIZER, START_TOKEN, END_TOKEN)

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

    def __init__(self, json_annotations_path, img_path='train', img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        self.json_path = pathlib.Path(json_annotations_path)
        self.image_path = pathlib.Path(TRAIN_IMGS_PATH if img_path =='train' else VAL_IMGS_PATH)
        self.image_width = img_width
        self.image_height = img_height

        data = json.loads(self.json_path.read_bytes())
        # extract only filenames and id
        # self.image_data = pl.DataFrame([dict(id=el['id'],
        #                         filename=el['file_name']) for el in data['images']])
        image_data_dict = dict(id=[element['id'] for element in data['images']],
                               filename=[element['file_name'] for element in data['images']])
        self.image_data = pl.DataFrame(image_data_dict)

        # extract only image_id and caption
        # self.annotations_data = pl.DataFrame([dict(id=el['image_id'],
        #                         caption=el['caption']) for el in data['annotations']])
        annotations_data_dict = dict(id = [element['image_id'] for element in data['annotations']],
                                     caption=[element['caption'] for element in data['annotations']])
        self.annotations_data = pl.DataFrame(annotations_data_dict)
        

    def __get_img_name_by_id__(self, image_id):
        try:
            image_name = self.image_data.filter(pl.col("id") == image_id).item(0, 1)
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
        captions = self.annotations_data.filter(pl.col('id') == image_id).select(pl.col('caption')).to_dict()['caption']
        return captions if len(captions) > 0 else None
    
    def __preprocessing__(self, captions):
        captions = captions.apply(lambda x: re.sub(r"[\.,:!?]",' ', x[0]))
        captions = captions.apply(lambda x: re.sub(r" +",' ', x[0]))
        captions = captions.select(pl.col("apply").str.to_lowercase())
        captions = captions.apply(lambda x: (START_TOKEN + " " + x[0] + END_TOKEN))
    
        # captions = captions.select(pl.col('caption').str.to_lowercase())
        return captions

    
    def build_vocabulary(self):

        def yield_token(captions):
            for example in captions.to_dict()['apply']:
                tokens = TOKENIZER(example)
                yield tokens

        # preprocess captions
        captions = self.annotations_data.select(pl.col('caption'))
        captions = self.__preprocessing__(captions)
        
        # creating vocabulary 
        token_generator = yield_token(captions)
        vocab = build_vocab_from_iterator(token_generator)
        vocab.set_default_index(-1)

        self.vocab = vocab
        
        
    def get_caption_tokens(self, only_unique=False, new_data: list[str] = None):
        """
        if only_unique is True, tokenize only one pair image-caption,
        otherwise, tokenize all captions, and image data will be repeated
        """
        if new_data is None:
            if only_unique:
                captions = self.image_data.apply(lambda x: self.get_caption_by_id(x[0])[0])
            else:
                captions = self.image_data.apply(lambda x: self.get_caption_by_id(x[0]))
                captions = captions.explode("apply")

        
        """
        captions = [[self.vocab[word] for word in list_word] for list_word in captions]
        captions = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in captions], batch_first=True)
        return caption
        """



def display_image(img, captions):
    for caption in captions:
        cv2.imshow(caption, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    json_val_name = ANN_PATH + '/captions_val2017.json'
    json_train_name = ANN_PATH + '/captions_train2017.json'
    parser = COCOparser(json_train_name, img_path='train')
    # img, captions = None , parser.get_caption_by_id(149)
    t = time.time() 
    parser.build_vocabulary()
    print(time.time() - t, '----- building vocabulary')
    t = time.time()
    parser.get_caption_tokens()
    print(time.time() - t, '----- tokenize')