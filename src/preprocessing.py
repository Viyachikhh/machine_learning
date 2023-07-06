
import cv2
import pathlib
import polars as pl
import re
import time
import torch

from typing import Union
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor

from consts import TOKENIZER, START_TOKEN, END_TOKEN, ANN_PATH, REGEX_VERBS
from coco_loader import COCOparser


def preprocess_image(self, image_path, resize=False):
        img = cv2.imread(image_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return img



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
        print("DATAFRAME")
        captions = data.select(pl.col('caption'))
        captions = captions.apply(lambda x: preprocess_text(x[0]))
        token_generator = yield_token_from_df(captions)
    else:
        print("LIST")
        captions = [preprocess_text(string) for string in data]
        token_generator = yield_token_from_list(captions)
    
    # creating vocabulary 
    vocab = build_vocab_from_iterator(token_generator)
    vocab.set_default_index(-1)

    return vocab
        
        
def get_tokens(vocab, data):
    preprocessed = [vocab(preprocess_text(element).split(' ')) for element in data]
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in preprocessed], batch_first=True)
        
    
    # captions = [[self.vocab[word] for word in list_word] for list_word in captions]
    # captions = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in captions], batch_first=True)
    # return caption

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
    t = time.time()
    example = get_tokens(vocab, annotations.to_dict()['caption'])
    print(time.time() - t, '----- tokenize full train', example.shape)

    # print(preprocess_text('My friend\'s horn'))