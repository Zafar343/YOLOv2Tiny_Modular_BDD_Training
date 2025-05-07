from multiprocessing import Process
from tqdm import tqdm
import os
import contextlib
import argparse
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
from dataset_config.factory import get_imdb
from dataset_config.roidb import RoiDataset, Custom_yolo_dataset, inference_collate
from yolo_eval import yolo_eval
from torch.utils.data import DataLoader
from config import config_bdd as cfg
import pascalvoc as pascalvoc
from util.data_util import check_dataset
import shutil
import warnings
from collections import OrderedDict
import cv2
import colorama
from colorama import Fore, Back, Style
from models_torch.torch_original import *
colorama.init(autoreset=True)
warnings.filterwarnings('ignore')

torch.manual_seed(0)
np.random.seed(0)

@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)

def parse_args():

    parser = argparse.ArgumentParser('Yolov2tiny')
    parser.add_argument('--dataset', dest='dataset',
                        default='custom', type=str)
    parser.add_argument('--data', type=str,
                        default="data.yaml", help='Give the path of custom data yaml file' )
    parser.add_argument('--output_dir', dest='output_dir',
                        default='eval', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='output_scratch/yolov2_best_map@347.pth',
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=8, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=True, action='store_true')
    parser.add_argument('--vis', dest='vis',
                        default=False, type=bool)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    # parser.add_argument('weights', type=str,
    #                     default='yolov2-pytorch/data/pretrained/yolov2-tiny-voc.pth',
    #                     help='model .pth path')
    parser.add_argument('--thres', type=float,
                       default=0.2, help='confidence threshold for selecting final predicitions')
    parser.add_argument('--pseudos', type=bool,
                        default=False, help='True if generating pseudo-labels')
    parser.add_argument('--device', default=0,
                        help='Choose a gpu device 0, 1, 2 etc.')
    parser.add_argument('--savePath', default='results')
    parser.add_argument('--imgSize', default='1280,720')
    parser.add_argument('--self_training', default=False)

    args = parser.parse_args()
    return args

def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info

def appendLists(a=[],b=[], im_info={}, thres=0.25, selfT=False):
    w = im_info['width'].item()
    h = im_info['height'].item()
    for i in range(len(b)):
        # if round(b[i][1],2) >= 0.2:
        # _smal_list = f'{int(b[i][0])} {round(b[i][1],2)} {round(b[i][2]/w,4)} {round(b[i][3]/h,4)} {round(b[i][4]/w,4)} {round(b[i][5]/h,4)} \n'
        if b[i][-1] > thres:
            width  = abs(b[i][3] - b[i][1])
            height = abs(b[i][4] - b[i][2])
            x_center = (b[i][3] + b[i][1]) / 2
            y_center = (b[i][4] + b[i][2]) / 2
            
            if not selfT:
                _smal_list = f'{int(b[i][0])} {round(b[i][-1],4)} {round(x_center/w, 4)} {round(y_center/h, 4)} {round(width/w, 4)} {round(height/h, 4)} \n'
            else:
                _smal_list = f'{int(b[i][0])} {round(x_center/w, 4)} {round(y_center/h, 4)} {round(width/w, 4)} {round(height/h, 4)} {round(b[i][-1],4)} \n'
            a.append(_smal_list)    
    return a

def util(check_point):
    dum = []
    for i, (k,v) in enumerate(check_point.items()):
        if k == 'conv9.0.weight':    #con9: torch.Size([40, 1024, 1, 1]), bias9: torch.Size([40])
            v = torch.rand((40, 1024, 1, 1))
            v /= 1000
            append = (k,v)
            dum.append(append)
        elif k == 'conv9.0.bias':
            v = torch.rand(40)
            v /= 10000
            append = (k,v)
            dum.append(append)
        else:
            append = (k,v)
            dum.append(append)
            # print(v)            
    modified_check_point = {"model": OrderedDict(dum)}
    return modified_check_point

def drawBox(label:np.array, img:np.ndarray, rel=False):
    # for i in range(label.shape[0]):
    h, w, _ = img.shape
    if label.size == 6:
        box = [label[1], label[2], label[3], label[4]]
    elif label.size == 5:
        box = [label[1], label[2], label[3], label[4]]    
    elif label.size == 4:    
        box = [label[0], label[1], label[2], label[3]]
    else:
        raise ValueError("Invalid size array only accept arrays of size 4 or 5")    
    
    color = list(np.random.random(size=3) * 256)
    if rel:
        img = cv2.rectangle(img,(int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), color, 3)
    else:
        img = cv2.rectangle(img,(int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
    return img

def showImg(img, labels, meta, relative=False):
    # Convert the tensor to a numpy array
    _image = img
    image_np = _image.numpy().transpose((1, 2, 0))
    # image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)*255
    _img = Image.fromarray(image_np.astype('uint8'), 'RGB')
    _img = np.array(_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
    _img = cv2.resize(_img, (int(meta['width'].item()),  int(meta['height'].item())), interpolation= cv2.INTER_LINEAR)
    for i in range(labels.shape[0]):
        label = labels[i].numpy()
        conf = label[-1]
        if conf >= 0.1:
            if relative:
                _img = drawBox(label, _img, True)
            else:    
                _img = drawBox(label, _img)
    cv2.imshow('', _img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def test(args, model, device=None):
    args.conf_thresh = 0.01
    args.nms_thresh = 0.4
    # if args.vis:
    #     args.conf_thresh = 0.4
    # device = int(args.device)
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device}'   
    print('Called with args:')
    print(args)
    
    if args.dataset == 'custom':
        data_dict = check_dataset(args.data)
        _, val_data, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
        nc = int(data_dict['nc'])  # number of classes
        names = data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {args.data}'  # check
    
    save_dir = os.path.join(args.output_dir, "preds")
    if not os.path.exists(save_dir):
        print(f'making: {Fore.GREEN}{save_dir}')
        os.makedirs(save_dir)
    else:
        print(f'{Fore.GREEN}{save_dir} {Fore.RESET}exists removing...')
        shutil.rmtree(f'{save_dir}', ignore_errors=True)
        print(f'making: {Fore.GREEN}{save_dir}')
        os.makedirs(save_dir)

    try:
        val_data
    except:
        val_data = None    
    
    if val_data is not None:
        # if args.withTrain:
        #     args.conf_thresh = 0.18
        #     args.nms_thresh = 0.35
        args.scale = True
        val_dataset  = Custom_yolo_dataset(data=val_data, train=False, cleaning = False)
        dataset_size = len(val_dataset)
        num_classes  = nc
        # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    else:
        args.scale = True
        args.dataset = "voc07test"
        num_classes  = 20
        # args.conf_thresh = 0.001
        # args.nms_thresh = 0.45
        # args.data_limit = 16
        # print(args)

        # prepare dataset

        if args.dataset      == 'voc07trainval':
            args.imdbval_name = 'voc_2007_trainval'

        elif args.dataset == 'voc07test':
            args.imdbval_name = 'voc_2007_test'

        else:
            raise NotImplementedError

        val_imdb = get_imdb(args.imdbval_name)

        val_dataset = RoiDataset(val_imdb, train=False)
        dataset_size = len(val_imdb.image_index)
        num_classes = val_imdb.num_classes
        all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    
    if not args.data_limit == 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(0, args.data_limit))

    # args.output_dir = args.temp_path
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # model = model    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                pin_memory=True,
                                num_workers=4)  
    
    if args.use_cuda:
        print(f"Validating using CUDA")

    args.output_dir = os.path.join(args.output_dir, "Outputs")
    os.makedirs( args.output_dir, exist_ok=True )
    if args.dataset!='custom':
        det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos, paths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Performing validation."):
            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)
            out, _, _ = model.Forward(im_data_variable)
            yolo_outputs = model.Forward_pred(out)
            for i in range(im_data.size(0)):
                # img = im_data
                img_id += 1
                if args.data is not None:
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'
                if args.dataset=='custom':
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh)
                if len(detections) > 0:
                    if args.dataset!='custom':
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_det = torch.zeros((inds.numel(), 5))
                                cls_det[:, :4] = detections[inds, :4]
                                cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                                showImg(im_data[i], cls_det, im_info)
                                all_boxes[cls][img_id] = cls_det.cpu().numpy()
                    elif args.dataset=='custom':
                        _detAllclass = []
                        for cls in range(num_classes):
                            inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                            if inds.numel() > 0:
                                cls_det = torch.zeros((inds.numel(), 6))
                                cls_det[:,0] = detections[inds, -1]
                                cls_det[:, 1:6] = detections[inds, :5]
                                _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                                _detAllclass = appendLists(_detAllclass, _det1Class, im_info, args.thres, args.self_training)
                        # if not os.path.exists(f'{save_dir}/labels'):
                        #     os.mkdir(f'{save_dir}/labels')
                        if len(_detAllclass)>0:
                            with open(f'{save_dir}/{name}', 'w') as f:
                                f.writelines(_detAllclass)                            
    if not args.self_training:
        if args.data is not None:
            args.gtFolder           =     val_dir
            args.detFolder          =     save_dir
            args.iouThreshold       =     args.nms_thresh
            args.gtFormat           =     'xywh'
            args.detFormat          =     'xywh'
            args.gtCoordinates      =     'rel'
            args.detCoordinates     =     'rel'
            args.imgSize            =     args.imgSize   # for bdd --> 1280, 720 and waymo --> 1920, 1280
            args.savePath           =     args.savePath
            args.call_with_train    =     False
            args.showPlot           =     False
            args.names              =     names
            args.val                =     True
            map, class_metrics = pascalvoc.main(args)
        return map, class_metrics   
    else:
        print(f'{Fore.GREEN} Detections saved in the designated folder for pseudo-label generation')
        return None, None

def test_for_train(temp_path, model, 
                   args, val_data=None, 
                   classes=None, 
                   afterTrain=False,
                   device=None):
    # make a directory to save predictions paths
    save_dir = f'{temp_path}/preds'
    if not os.path.exists(save_dir):
        print(f'making: {Fore.GREEN}{save_dir}')
        os.makedirs(save_dir)
    else:
        print(f'{Fore.GREEN}{save_dir} {Fore.RESET}already exists removing...')
        shutil.rmtree(f'{save_dir}', ignore_errors=True)
        print(f'making: {Fore.GREEN}{save_dir}')
        os.makedirs(save_dir)

    conf_thresh = 0.01
    nms_thresh = 0.4
    thres = 0.25
    scale = True
    val_dataset = Custom_yolo_dataset(data=val_data, train=False, cleaning=False)
    dataset_size = len(val_dataset)
    num_classes = len(classes)
    # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(num_classes)]
    
    if not args.data_limit==0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(0, args.data_limit))

    args.output_dir = temp_path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    model = model
    
    # with num_torch_thread(1):    
    val_dataloader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=inference_collate)

    if args.use_cuda:  
        print(f"Validating using CUDA")

    args.output_dir = os.path.join(args.output_dir, "Outputs")
    os.makedirs( args.output_dir, exist_ok=True )
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos, paths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Performing validation."):
            im_data_variable = Variable(im_data).cuda()
            out, _, _ = model.Forward(im_data_variable)
            yolo_outputs = model.Forward_pred(out)
            for i in range(im_data.size(0)):
                img_id += 1
                if args.data is not None:
                    name = paths[i].split('/')[-1]
                    name = name.split('.')[0] + '.txt'                
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=conf_thresh,
                                        nms_threshold=nms_thresh)
                if len(detections) > 0:
                    _detAllclass = []
                    for cls in range(num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 6))
                            cls_det[:,0] = detections[inds, -1]
                            cls_det[:, 1:6] = detections[inds, :5]
                            _det1Class = cls_det.tolist()         # per class detections tensor of (N,6) [cls conf x y w h]
                            _detAllclass = appendLists(_detAllclass, _det1Class, im_info, thres)
                    if len(_detAllclass)>0:
                        with open(f'{save_dir}/{name}', 'w') as f:
                            f.writelines(_detAllclass)                                    
    
    if args.data is not None:
        args.gtFolder           =   args.val_dir
        args.detFolder          =   save_dir
        args.iouThreshold       =   nms_thresh
        args.gtFormat           =   'xywh'
        args.detFormat          =   'xywh'
        args.gtCoordinates      =   'rel'
        args.detCoordinates     =   'rel'
        args.imgSize            =   args.imgSize   # for bdd --> 1280, 720 and waymo --> 1920, 1280
        args.savePath           =   args.savePath
        args.call_with_train    =   True
        args.showPlot           =   False
        args.names              =   classes
        args.val                =   afterTrain
        map, class_metrics = pascalvoc.main(args)    
        
    return map, class_metrics


if __name__ == '__main__':
    model = DeepConvNetTorch(input_dims=(3, 416, 416),
                                    num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                    max_pools=[0, 1, 2, 3, 4],
                                    weight_scale='kaiming',
                                    batchnorm=True,
                                    num_classes=1,
                                    dtype=torch.float32, device='cuda')

    args = parse_args()
    # modelPath = 'output/yolov2_best_map@211.pth'
    checkpoint = torch.load(args.model_name)
    pytorch_model = checkpoint['model']

    for param, val in model.params.items():
        for param1, val1 in pytorch_model.items():
            if (param == param1):
                model.params[param] = val1.cuda()
    
    map, metrics = test(args, model)