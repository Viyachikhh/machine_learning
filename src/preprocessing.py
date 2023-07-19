
import pathlib
import polars as pl
import re
import warnings
import time
import torch
import torchvision.transforms.v2 as transforms

from cv2 import imread, resize, INTER_AREA
from typing import Union
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor
from torchvision.io import read_image

from src.consts import (TOKENIZER, START_TOKEN, END_TOKEN, ANN_PATH, REGEX_VERBS, 
                    DEVICE, IMAGE_RESIZE_OBJ, IMAGE_HEIGHT, IMAGE_WIDTH)
from src.coco_loader import COCOparser

warnings.filterwarnings("ignore")


def preprocess_image(image_path, resize_obj=IMAGE_RESIZE_OBJ):
        img = read_image(image_path)
        img = resize_obj(img) / 255.
        return img


def preprocess_text(string):
    """
    Cleaning string from punkt and letters
    """
    prep_string = re.sub(r"[\.,:!?\"]+",' ', string)
    # prep_string = re.sub(REGEX_VERBS, ' ', prep_string)
    prep_string = START_TOKEN + " " + prep_string + " " + END_TOKEN
    prep_string = re.sub(r" +",' ', prep_string).lower()
    return prep_string

def build_vocabulary(data: Union[pl.DataFrame, list[str]]):
    """
    Create voabulary from pl.DataFrame or list of str
    """

    def yield_token_from_df(captions):
        for example in captions.to_dict()['apply']:
            tokens = TOKENIZER(example)
            yield tokens

    def yield_token_from_list(captions):
        for example in captions:
            tokens = TOKENIZER(example)
            yield tokens

    # preprocess captions
    if isinstance(data, pl.DataFrame):
        captions = data.select(pl.col('caption'))
        captions = captions.apply(lambda x: preprocess_text(x[0]))
        token_generator = yield_token_from_df(captions)
    else:
        captions = [preprocess_text(string) for string in data]
        token_generator = yield_token_from_list(captions)
    
    # creating vocabulary 
    vocab = build_vocab_from_iterator(token_generator)
    vocab.set_default_index(-1)

    return vocab
                
def get_tokens(vocab, data: list[str]) -> torch.Tensor:
    """
    [['a very good day ...'],...] -> [[2, 0, 152, 13, ...],...]
    """
    preprocessed = [vocab(preprocess_text(element).split(' ')) for element in data]
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in preprocessed], batch_first=True)


def extract_filename_and_caption(loader, unique_images=False):
    image_folder = loader.image_path.as_posix()
    img_df = loader.get_image_data
    captions_df = loader.get_annotations_data
    info = img_df.join(captions_df, on='id').select([pl.col('filename'), pl.col('caption')])
    if unique_images:
        info = info.unique(subset=['filename'], keep='first')
    info = info.apply(lambda x: (image_folder + '/' + x[0], x[1]))   # .rename({'column_0':'path', 'column_1': 'caption'})
    return info

def generator_from_dataframe(df, vocab, batch_size = 32):
    batch_count = df.shape[0] // batch_size
    anns = df.select(pl.col('column_1')).to_dict()['column_1']
    tokens = get_tokens(vocab, anns)
    for i in range(batch_count):
        part = df.slice(i * batch_size, batch_size)
        img_path = part.select(pl.col('column_0')).to_dict()['column_0']
        imgs = torch.cat([preprocess_image(el) for el in img_path]).to(DEVICE, dtype=torch.float16)
        left_ind, right_ind = i * batch_size, (i + 1) * batch_size
        text = (tokens[left_ind: right_ind]).to(DEVICE, dtype=torch.int32)
        yield imgs, text, left_ind, right_ind



if __name__ == "__main__":
    json_val_name = ANN_PATH + '/captions_val2017.json'
    json_train_name = ANN_PATH + '/captions_train2017.json'
    loader = COCOparser(json_train_name, img_path='train')
    # img, captions = loader.get_img_by_id(139) , loader.get_caption_by_id(139)
    # print(img, captions)

    t = time.time()
    annotations =  loader.get_annotations_data
    vocab = build_vocabulary(annotations.to_dict()['caption'])
    print(time.time() - t, '----- building vocabulary')
    # t = time.time()
    # example = get_tokens(vocab, annotations.to_dict()['caption'])
    # print(time.time() - t, '----- tokenize full train', example.shape)
    t = time.time()
    df = extract_filename_and_caption(loader, unique_images=True)
    for el in generator_from_dataframe(df, vocab):
        print(el)
    