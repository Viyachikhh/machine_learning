
import pathlib
import polars as pl
import re
import time
import torch
import torchvision.transforms.v2 as transforms

from typing import Union
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor
from torchvision.io import read_image

from consts import TOKENIZER, START_TOKEN, END_TOKEN, ANN_PATH, REGEX_VERBS, DEVICE
from coco_loader import COCOparser


def preprocess_image(image_path, resize_obj):
        img = read_image(image_path.as_posix())
        return resize_obj(img)

def preprocess_text(string):
    prep_string = re.sub(r"[\.,:!?\"]+",' ', string)
    prep_string = re.sub(REGEX_VERBS, ' ', prep_string)
    prep_string = re.sub(r" +",' ', prep_string).lower()
    prep_string = START_TOKEN + " " + prep_string + " " + END_TOKEN
    return prep_string

def build_vocabulary(data: Union[pl.DataFrame, list[str]]):

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
                
def get_tokens(vocab, data):
    preprocessed = [vocab(preprocess_text(element).split(' ')) for element in data]
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in preprocessed], batch_first=True)


def extract_filename_and_caption(loader, unique_images=False):
    img_df = loader.get_image_data
    captions_df = loader.get_annotations_data
    info = img_df.join(captions_df, on='id').select([pl.col('filename'), pl.col('caption')])
    if unique_images:
        info = info.unique(subset=['filename'], keep='first')
    return info


if __name__ == "__main__":
    json_val_name = ANN_PATH + '/captions_val2017.json'
    json_train_name = ANN_PATH + '/captions_train2017.json'
    loader = COCOparser(json_train_name, img_path='train')
    # img, captions = loader.get_img_by_id(139) , loader.get_caption_by_id(139)
    # print(img, captions)

    # t = time.time()
    # annotations =  loader.get_annotations_data
    # vocab = build_vocabulary(annotations.to_dict()['caption'])
    # print(time.time() - t, '----- building vocabulary')
    # t = time.time()
    # example = get_tokens(vocab, annotations.to_dict()['caption'])
    # print(time.time() - t, '----- tokenize full train', example.shape)
    print(extract_filename_and_caption(loader))

    