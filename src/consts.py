import re
from torchtext.data.utils import get_tokenizer


IMAGE_HEIGHT = 448
IMAGE_WIDTH = 448

ORIGINAL_DATASET_PATH = '/home/viyachikhh/Datasets/coco'

TRAIN_IMGS_PATH = ORIGINAL_DATASET_PATH + '/coco_train2017'
VAL_IMGS_PATH = ORIGINAL_DATASET_PATH + '/coco_val2017'
TEST_IMGS_PATH = ORIGINAL_DATASET_PATH + '/coco_test2017'
ANN_PATH = ORIGINAL_DATASET_PATH + '/coco_ann2017'

STOPWORDS = ["a", "an", "the", "this", "that", "is", "it", "to", "and"]

TOKENIZER = get_tokenizer('basic_english')

START_TOKEN = '<|startoftext|>'
END_TOKEN = '<|endoftext|>'
REGEX_VERBS = r"\'re |\'ve |\'ll |\'s |\'m |\'d "