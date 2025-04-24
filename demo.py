from tqdm import tqdm
import os
import glob
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from yolo_eval import yolo_eval
from config import config_bdd as cfg
import shutil
import warnings
import cv2
import colorama
from colorama import Fore, Back, Style
from models_torch.torch_original import *
colorama.init(autoreset=True)
warnings.filterwarnings('ignore')

np.random.seed(0)

def demo_args():
    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='demo_output', type=str,
                        help='Specify an output directory')
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_best_map.pth', type=str,
                        help='Specify the model path (.pth model)')
    parser.add_argument('--use_cuda', dest='use_cuda',
                        default=True, type=bool,
                        help='specify whether to use cuda or not')
    parser.add_argument('--data', type=str,
                        default=None,
                        help='Path to data dir or data.txt file')
    parser.add_argument('--classes', nargs="*",
                        default=["vehicles"],
                        help='provide the list of class names if other than voc') # Example 'default = ['Vehicle'] for a single class
    parser.add_argument('--conf_thresh', type=float,
                        default=0.1,
                        help='choose a confidence threshold for inference')
    parser.add_argument('--nms_thresh', type=float,
                        default=0.45,
                        help='choose an nms threshold for post processing predictions, must be in range 0-1s')
    parser.add_argument('--thresh', type=float,
                        default=0.5,
                        help='choose a threshold for inference')
    parser.add_argument('--device', type=int,
                        default=0,
                        help='choose a gpu device for cuda inference')
    parser.add_argument('--save-txt', type=bool,
                        default=False,
                        help='save predictions to a file')
    parser.add_argument('--save-annotated-img', type=bool,
                        default=False,
                        help='save predictions to a file')
    parser.add_argument('--vis', type=bool,
                        default=False,
                        help='save predictions to a file')
    args, unknown = parser.parse_known_args()
    return args

def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- cv.image

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """
    
    im_info = dict()
    im_info['height'], im_info['width'] , _ = img.shape

    # resize the image
    H, W = cfg.input_size
    img = cv2.resize(img, (W,H))
    im_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im_data = torch.from_numpy(im_data).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info

def drawBox(label:np.array, img:np.ndarray, classes, rel=False):
    # for i in range(label.shape[0]):
    h, w, _ = img.shape
    if label.size == 6:
        box = [label[0], label[1], label[2], label[3]]
        cls =  classes[int(label[-1])]
        conf = round(label[-2], 2)
    elif label.size == 5:
        box = [label[0], label[1], label[2], label[3]]
        conf = label[-1]    
    elif label.size == 4:    
        box = [label[0], label[1], label[2], label[3]]
    else:
        raise ValueError("Invalid size array only accept arrays of size 4 or 5")    
    
    color = [(0,0,255), (0,255,0), (255,0,0)]
    if not cls:
        cls = ''
    if not conf:
        conf = ''    
    text = f"{cls} {conf}:"
    if rel:
        img = cv2.rectangle(img,(int(box[0]*w), int(box[1]*h)), 
                            (int(box[2]*w), int(box[3]*h)), 
                            color[int(label[-1])], 2)
        
        cv2.putText(img, text, (int((box[0]*w)+2), int((box[1]*h) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[int(label[-1])], 1)
    else:
        img = cv2.rectangle(img,(int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            color[int(label[-1])], 2)
        
        cv2.putText(img, text, (int((box[0])+2), int((box[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    color[int(label[-1])], 1)
    return img

def showImg(img, labels, meta=False, cls='', conf_thres=0.01, relative=False):
    # Convert the tensor to a numpy array
    # _img = img
    # _img = np.array(_img)
    if meta:
        _img = cv2.resize(img, (int(meta['width'].item()),  int(meta['height'].item())), interpolation= cv2.INTER_LINEAR)
    for i in range(labels.shape[0]):
        label = labels[i]
        conf = label[-2]
        if conf >= conf_thres:
            if relative:
                _img = drawBox(label, img, True)
            else:    
                _img = drawBox(label, img, cls)
    return _img

def demo(args):
    print('Inference call with args: {}'.format(args))     
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.data==None:
        # input images
        images_dir   = 'Images'
        images_names = ['image1.jpg', 'image2.jpg']
    else:
        if os.path.isdir(args.data):
            images_names = glob.glob(f"{args.data}/*.jpg")
        else:    
            with open(args.data, 'r') as f:
                images_names = f.readlines()
    
    try: args.classes
    except: args.classes=None
    
    if args.classes is None:
        classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor')
    else:
        classes = args.classes

    # set the save folder path for predictions
    if args.data==None:
        pred_dir = os.path.join(images_dir, 'preds')
        pred_dir = os.path.join(os.getcwd(),pred_dir)    
    else:
        pred_dir = args.output_dir                    
    
    if not os.path.exists(pred_dir):
        print(f'making {pred_dir}')
        os.makedirs(pred_dir)
    else:
        print('Deleting existing pred dir')
        shutil.rmtree(pred_dir, ignore_errors=True)
        print(f'making new {pred_dir}')
        os.makedirs(pred_dir)

    model = DeepConvNetTorch(input_dims=(3, 416, 416),
                                    num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                    max_pools=[0, 1, 2, 3, 4],
                                    weight_scale='kaiming',
                                    batchnorm=True,
                                    num_classes=len(classes),
                                    dtype=torch.float32, device='cuda')           

    checkpoint = torch.load(args.model_name)
    pytorch_model = checkpoint['model']

    for param, val in model.params.items():
        for param1, val1 in pytorch_model.items():
            if (param == param1):
                model.params[param] = val1.cuda()
    # model.eval()

    for image_name in tqdm(images_names):
        if args.data==None:
            image_path = os.path.join(images_dir, image_name)
            # img        = Image.open(image_path)
            img = cv2.imread(image_path, 3)
        else:
            image_path = image_name.strip()
            # img        = Image.open(image_path)
            img = cv2.imread(image_path, 3)

        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            # im_data_variable = Variable(im_data).cuda()
            im_data_variable = Variable(im_data).to(device)
        else:
            im_data_variable = Variable(im_data)    
        
        out, _, _ = model.Forward(im_data_variable)
        yolo_outputs = model.Forward_pred(out)
        output = [item[0].data for item in yolo_outputs]

        detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh, nms_threshold=args.nms_thresh)
        if len(detections) > 0:
            name        = image_name.split('/')[-1]
            pred_name   = name.split('.')[0] + '.txt'
            pred_path   = os.path.join(pred_dir, pred_name)
            det_boxes   = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            pred        = np.zeros((det_boxes.shape[0],6))
            pred[:, :5] = det_boxes
            pred[:,-1]  = det_classes # pred is [x, y, x, y, conf, cls]
            
            pred = pred[pred[:,-2]>args.thresh]

            if len(pred) > 0:
                if args.save_txt:
                    _detAllclass = []
                    for _pred in pred:
                        _detAllclass.append(f"{int(_pred[-1])} {_pred[0]:.4f} {_pred[1]:.4f} {_pred[2]:.4f} {_pred[3]:.4f} {_pred[4]:.4f}\n")
                with open(pred_path, 'w') as f:
                    f.writelines(_detAllclass)

                if args.save_annotated_img: 
                    img = showImg(img, pred, cls=classes)
                    cv2.imwrite(pred_path.replace('.txt', '.jpg'), img)
                
                if args.vis:
                    img = showImg(img, pred, cls=classes)
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = demo_args()
    demo(args)

