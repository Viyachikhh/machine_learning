import torch
import sys
import os
from src.models import TransformLayer, TextTransformer, VisionTransformer

def check_availability_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    input_1 = torch.randn(1000,53, 768)
    # transform_layer = TransformLayer(768, 12)
    # res_1 = transform_layer(input_1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_2 = torch.randn(100,3,224,224).to(device)
    vision = VisionTransformer(224, 64, 4, 2, 12, 112).to(device)
    #res_2 = vision(input_2)
    #print(res_2.shape)
    # param_size = 0
    # for param in vision.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in vision.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    # size_all_mb = (param_size + buffer_size) / 1024**2
    print(vision(input_2).shape)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""