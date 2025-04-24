import numpy as np
import pickle
import torch
from dataset_config.factory import *
from dataset_config.roidb import *
from dataset_config.imdb import *
from dataset_config.pascal_voc import *
from dataset_config.yolo_eval import *
from Dataset.factory import *
from Dataset.roidb import *
from Dataset.imdb import *
from Dataset.pascal_voc import *
from Dataset.yolo_eval import *
from models.torch_original import *
import random

torch.manual_seed(42)
random.seed(0)
np.random.seed(42)
# with open('../weight_2bn/set2/epoch_120.pkl', 'rb') as f:
#     x = pickle.load(f)
# model = x['model']

model = DeepConvNetTorch(input_dims=(3, 416, 416),
                                 num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                 max_pools=[0, 1, 2, 3, 4],
                                 weight_scale='kaiming',
                                 batchnorm=True,
                                 dtype=torch.float32, device='cpu')

checkpoint = torch.load('./Dataset/yolov2_epoch_299.pth')
pytorch_model = checkpoint['model']

for param, val in model.params.items():
    for param1, val1 in pytorch_model.items():
        if (param == param1):
            model.params[param] = val1.cpu().detach()
            # if model.params[param].ndim == 1:
            #     print(model.params[param][0:5])
            # elif model.params[param].ndim == 4:
            #     print(model.params[param][0][0][0:10][0])

with open("../test_folder/yolov2tiny_inference/Dataset/dummy_data.pkl", 'rb') as f:
    dummy_data = pickle.load(f)

dummy_data = iter(dummy_data)

im_data, gt_boxes, gt_classes, num_obj = next(dummy_data)
print(gt_boxes[0:10])

# f1 = time.time()
out, cache, FOut = model.Forward(im_data)
# f2 = time.time()
# print(f"Forward time: {f2-f1:.4f}")
loss, loss_grad = model.loss(out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)
print(loss, loss_grad.shape)
# f3 = time.time()
# # print(f"Calculate loss time: {f3 - f2:.4f}")
lDout, grads = model.backward(loss_grad, cache)
# f4 = time.time()
# print(f"Backward time: {f4 - f3:.4f}")
