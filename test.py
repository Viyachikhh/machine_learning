import gc
import torch
import sys
import os
import pytest


from src.models import TextTransformer, VisionTransformer, convert_weights_to_fp16
from src.coco_loader import COCOparser
from src.preprocessing import extract_filename_and_caption, generator_from_dataframe, build_vocabulary
from src.consts import *

def check_availability_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


def test_loader():
    json_val_name = ANN_PATH + '/captions_val2017.json'
    validation_loader = COCOparser(json_annotations_path=json_val_name)
    assert not validation_loader.get_annotations_data.is_empty()
    assert not validation_loader.get_image_data.is_empty()


def test_extracting_and_preprocessing():
    json_val_name = ANN_PATH + '/captions_val2017.json'
    validation_loader = COCOparser(json_annotations_path=json_val_name, img_path='val')
    
    df_1 = extract_filename_and_caption(validation_loader)
    assert df_1.shape[0] == 25014

    df_2 = extract_filename_and_caption(validation_loader, unique_images=True)
    assert df_2.shape[0] == 5000

    annotations = validation_loader.get_annotations_data
    vocab = build_vocabulary(annotations.to_dict()['caption'])
    for chunk in generator_from_dataframe(df_2, vocab):
        assert chunk[0].dtype == torch.float16
        assert chunk[1].dtype == torch.int32
        assert (chunk[3] - chunk[2]) == 32
        break


def test_vision_architecture():
    batch_size = 1000
    output_dim = 112
    width = 64
    img_res = 224
    patch_size = 14
    layers_count = 2
    n_heads = 8
    assert width % n_heads == 0
    vision = VisionTransformer(img_res, width, patch_size, layers_count, n_heads, output_dim).cuda()
    convert_weights_to_fp16(vision)
    input_ = torch.randn(batch_size, 3, img_res, img_res).cuda()
    print(vision(input_.type(vision.dtype)).shape)
    # assert vision(input_).shape 


def test_text_architecture():
    pass

if __name__ == '__main__':
    test_vision_architecture()
    gc.collect()

