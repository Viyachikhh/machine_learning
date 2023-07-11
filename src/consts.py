import re
import warnings
from torch import device
from torch.cuda import is_available
from torch.nn import Sequential
from torchtext.data.utils import get_tokenizer
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

warnings.filterwarnings("ignore")


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

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

DEVICE = device('cuda' if is_available() else 'cpu')

IMAGE_RESIZE_OBJ = Compose([Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH))])
