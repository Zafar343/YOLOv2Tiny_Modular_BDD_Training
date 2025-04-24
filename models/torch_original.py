import torch
import torch.nn as nn
import numpy as npp
from autograd import grad, elementwise_grad
# import autograd.numpy as npauto
import autograd.numpy as np

import random
import torch.nn.functional as F
import shutil
import os
import warnings
import torch
import sys
from config import config_bdd as cfg
import pickle
import mygrad as mg
warnings.simplefilter("ignore", UserWarning)

def pytorch_to_mygrad(pytorch_tensor):
    data = pytorch_tensor.detach().cpu().numpy()
    
    mygrad_tensor = mg.Tensor(data)
    
    if pytorch_tensor.grad is not None:
        grad = pytorch_tensor.grad.cpu().numpy()
        mygrad_tensor.backward(grad)
    
    return mygrad_tensor

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def to_torch(x):
    return torch.from_numpy(x).float()

def mse_loss_numpy(input, target, reduction='sum', axis=None):

    if input.shape != target.shape:
        raise ValueError("Input and target shapes must be the same.")
    
    squared_diff = np.square(input - target)
    
    if reduction == 'sum':
        return np.sum(squared_diff, axis=axis)
    else:
        raise ValueError("Only 'sum' reduction is implemented.")
    
# def cross_entropy_numpy(input, target):
#     m = input.shape[0]
#     p = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
#     log_likelihood = -np.log(p[range(m), target])
#     return np.sum(log_likelihood) / m

def cross_entropy_numpy(class_scores, class_targets, epsilon=1e-12):
    """
    Compute cross entropy loss.
    
    Args:
    class_scores: numpy array of shape (N, C) where N is the number of samples
                  and C is the number of classes. Contains the raw class scores.
    class_targets: numpy array of shape (N,) containing the true class labels.
    epsilon: small value to avoid log(0)
    
    Returns:
    loss: scalar, sum of cross entropy loss over all samples
    """
    N = class_scores.shape[0]
    
    # Compute softmax probabilities
    exp_scores = np.exp(class_scores - np.max(class_scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute cross entropy loss
    correct_logprobs = -np.log(probs[range(N), class_targets] + epsilon)
    
    # Sum the loss over all samples
    loss = np.sum(correct_logprobs)
    
    return loss

def compute_gradients(loss_func, *args):
    return grad(loss_func)(*args)

def compute_loss(output_variable, target_variable):
# Compute the individual loss components
    loss = yolo_loss(output_variable, target_variable)

# For simplicity, we're using a single loss here instead of separate box, IOU, and class losses

    return loss

# Example usage:
# input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
# target_array = np.array([[2.0, 2.0], [4.0, 3.0]])
# loss = mse_loss_numpy(input_array, target_array, axis=1)

class DeepConvNetTorch(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and 
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:
    
    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a Torch_ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self, input_dims=(3, 32, 32),
                            num_filters=[8, 8, 8, 8, 8],
                            max_pools=[0, 1, 2, 3, 4],
                            batchnorm=False,
                            slowpool=True,
                            num_classes=10, weight_scale=1e-3, reg=0.0,
                            weight_initializer=None,
                            dtype=torch.float64, device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of convolutional
            filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro layers that
            should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random initialization
            of weights, or the string "kaiming" to use Kaiming initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization should
            only be applied to convolutional and fully-connected weight matrices;
            it should not be applied to biases or to batchnorm scale and shifts.
        - dtype: A torch data type object; all computations will be performed using
            this datatype. float is faster but less accurate, so you should use
            double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'    
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype
        self.slowpool = slowpool
        self.num_filters = num_filters
    
        if device == 'cuda':
            device = 'cuda:0'
        

        filter_size = 3
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pred_filters,H_out,W_out = input_dims
        HH = filter_size
        WW =  filter_size
        for i,num_filter in enumerate(num_filters):
            i += 1
            H_out = int(1 + (H_out + 2 * conv_param['pad'] - HH) / conv_param['stride'])
            W_out = int(1 + (W_out + 2 * conv_param['pad'] - WW) / conv_param['stride'])
            if self.batchnorm:
                self.params['bn{}.running_mean'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.params['bn{}.running_var'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.params['bn{}.weight'.format(i)] =0.01*torch.randn(num_filter, device =device, dtype = dtype)
                self.params['bn{}.bias'.format(i)] = 0.01*torch.randn(num_filter, device =device, dtype = dtype)
            if i in max_pools:
                H_out = int(1 + (H_out - pool_param['pool_height']) / pool_param['stride'])
                W_out = int(1 + (W_out - pool_param['pool_width']) / pool_param['stride'])
            if weight_scale == 'kaiming':
                self.params['conv{}.weight'.format(i)] = kaiming_initializer(num_filter, pred_filters, K=filter_size, relu=True, device=device,dtype=dtype)
            else:
                self.params['conv{}.weight'.format(i)] = torch.zeros(num_filter,pred_filters, filter_size,filter_size, dtype=dtype,device = device)
                self.params['conv{}.weight'.format(i)] += weight_scale*torch.randn(num_filter,pred_filters, filter_size,filter_size, dtype=dtype,device= device)
            pred_filters = num_filter

        i+=1
        # if weight_scale == 'kaiming':
        #     self.params['W{}'.format(i)] = kaiming_initializer(num_filter*H_out*W_out, num_classes, relu=False, device=device,dtype=dtype)
        # else:
        #     self.params['W{}'.format(i)] = torch.zeros(num_filter*H_out*W_out, num_classes, dtype=dtype,device = device)
        #     self.params['W{}'.format(i)] += weight_scale*torch.randn(num_filter*H_out*W_out, num_classes, dtype=dtype,device= device)
        # self.params['b{}'.format(i)] = torch.zeros(num_classes, dtype=dtype,device= device)
        # print(i)
        if weight_scale == 'kaiming':
                self.params['conv{}.0.weight'.format(i)] = kaiming_initializer(125, 1024, K=1, relu=False, device=device,dtype=dtype)
        # else:
        #     self.params['W{}'.format(i)] = torch.zeros(num_filter*H_out*W_out, num_classes, dtype=dtype,device = device)
        #     self.params['W{}'.format(i)] += weight_scale*torch.randn(num_filter*H_out*W_out, num_classes, dtype=dtype,device= device)
        self.params['conv{}.0.bias'.format(i)] = torch.zeros(125, dtype=dtype,device= device)

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the Forward pass
        # of the first batch normalization layer, self.bn_params[1] to the Forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
            for i, num_filter in enumerate(num_filters):
                self.bn_params[i]['bn{}.running_mean'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.bn_params[i]['bn{}.running_var'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 3  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        # assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batchnorm,
            'bn_params': self.bn_params,
        }
            
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']


        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))


    def train(self, X, gt_boxes=None, gt_classes=None, num_boxes=None):
        Forward_prop = True
        cal_loss = True
        backward_prop = True
        
        if Forward_prop:
            out,   cache = self.Forward(X)
            with open('Temp_Files/Pytorch_Forward_Out.pickle','wb') as handle:
                pickle.dump(out,handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('Temp_Files/Pytorch_Forward_Cache.pickle','wb') as handle:
                pickle.dump(cache,handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('Temp_Files/Pytorch_Forward_Out.pickle', 'rb') as handle:
                out = pickle.load(handle)
                out.requires_grad = True
                out.retain_grad()
            with open('Temp_Files/Pytorch_Forward_Cache.pickle', 'rb') as handle:
                cache = pickle.load(handle)
    
        if cal_loss:
            loss,   loss_grad = self.loss(out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_boxes)
            with open('Temp_Files/Pytorch_loss.pickle','wb') as handle:
                pickle.dump(loss,handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('Temp_Files/Pytorch_loss_grad.pickle','wb') as handle:
                pickle.dump(loss_grad,handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('Temp_Files/Pytorch_loss.pickle', 'rb') as handle:
                loss = pickle.load(handle)
            with open('Temp_Files/Pytorch_loss_grad.pickle', 'rb') as handle:
                loss_grad = pickle.load(handle)
                
        if backward_prop:   
            lDout, grads = self.backward(loss_grad, cache)
            with open('Temp_Files/Pytorch_Backward_lDout.pickle','wb') as handle:
                pickle.dump(lDout,handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('Temp_Files/Pytorch_Backward_grads.pickle','wb') as handle:
                pickle.dump(grads,handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('Temp_Files/Pytorch_Backward_lDout.pickle', 'rb') as handle:
                lDout = pickle.load(handle)
            with open('Temp_Files/Pytorch_Backward_grads.pickle', 'rb') as handle:
                grads = pickle.load(handle)

        return out, cache, loss, loss_grad, lDout, grads
    
    def Forward(self, X, gt_boxes=None, gt_classes=None, num_boxes=None):
        """
        Evaluate loss and gradient for the deep convolutional network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        y = 1
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        # pass conv_param to the Forward pass for the convolutional layer
        # Torch_Padding and stride chosen to preserve the input spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the Forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        slowpool_param = {'pool_height':2, 'pool_width':2, 'stride': 1}
        cache = {}
        out = X
        out,cache['0'] = Torch_Conv_BatchNorm_ReLU_Pool.Forward(out, self.params['conv1.weight'], self.params['bn1.weight'], self.params['bn1.bias'], conv_param, self.bn_params[0],pool_param)
        out,cache['1'] = Torch_Conv_BatchNorm_ReLU_Pool.Forward(out, self.params['conv2.weight'], self.params['bn2.weight'], self.params['bn2.bias'], conv_param, self.bn_params[1],pool_param)
        out,cache['2'] = Torch_Conv_BatchNorm_ReLU_Pool.Forward(out, self.params['conv3.weight'], self.params['bn3.weight'], self.params['bn3.bias'], conv_param, self.bn_params[2],pool_param)
        out,cache['3'] = Torch_Conv_BatchNorm_ReLU_Pool.Forward(out, self.params['conv4.weight'], self.params['bn4.weight'], self.params['bn4.bias'], conv_param, self.bn_params[3],pool_param)
        out,cache['4'] = Torch_Conv_BatchNorm_ReLU_Pool.Forward(out, self.params['conv5.weight'], self.params['bn5.weight'], self.params['bn5.bias'], conv_param, self.bn_params[4],pool_param)
        out,cache['5'] = Torch_Conv_BatchNorm_ReLU.Forward     (out, self.params['conv6.weight'], self.params['bn6.weight'], self.params['bn6.bias'], conv_param, self.bn_params[5]) 
        out            = F.pad                                 (out, (0, 1, 0, 1))
        out,cache['60']= Torch_FastMaxPool.Forward             (out, slowpool_param)
        out,cache['6'] = Torch_Conv_BatchNorm_ReLU.Forward     (out, self.params['conv7.weight'], self.params['bn7.weight'], self.params['bn7.bias'], conv_param, self.bn_params[6]) 
        out,cache['7'] = Torch_Conv_BatchNorm_ReLU.Forward     (out, self.params['conv8.weight'], self.params['bn8.weight'], self.params['bn8.bias'], conv_param, self.bn_params[7]) 
        conv_param['pad'] = 0
        out,cache['8'] = Torch_FastConvWB.Forward              (out, self.params['conv9.0.weight'], self.params['conv9.0.bias'], conv_param)
        
        return out, cache, out
    

    def Forward_pred(self, out, num_anchors=5, num_classes=1):
            """
            Evaluate loss and gradient for the deep convolutional network.
            Input / output: Same API as ThreeLayerConvNet.
            """
            # print('Calculating the loss and its gradients for pytorch model.')


            scores = out
            bsize, _, h, w = out.shape
            # out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)
            out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * num_anchors, 5 + num_classes)
            # Calculate losses based on loss functions(box loss, Intersection over Union(IoU) loss, class loss)
            xy_pred = torch.sigmoid(out[:, :, 0:2]) #
            conf_pred = torch.sigmoid(out[:, :, 4:5]) # 
            hw_pred = torch.exp(out[:, :, 2:4])
            class_score = out[:, :, 5:]
            class_pred = F.softmax(class_score, dim=-1)
            delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

            # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
            # dout = pickle.load(dout)
            # print('\n\n',dout.dtype, dout[dout!=0])
            return delta_pred, conf_pred, class_pred

    def loss(self, out, gt_boxes=None, gt_classes=None, num_boxes=None, num_anchors=5, num_classes=1):
        """
        Evaluate loss and gradient for the deep convolutional network.
        Input / output: Same API as ThreeLayerConvNet.
        """

        # print('Calculating the loss and its gradients for pytorch model.')
        out = pytorch_to_mygrad(out)
        scores = out
        bsize, _, h, w = out.shape
        out = out.transpose(0, 2, 3, 1).reshape(bsize, h * w * num_anchors, 5 + num_classes)
        # out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

        xy_pred = sigmoid(out[:, :, 0:2])
        conf_pred = sigmoid(out[:, :, 4:5])
        hw_pred = np.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        # class_pred = np.softmax(class_score, axis=-1)
        delta_pred = mg.concatenate([xy_pred, hw_pred], axis=-1)


        output_variable = (delta_pred, conf_pred, class_score)
        output_data = [v.data for v in output_variable]
        gt_data = (gt_boxes, gt_classes, num_boxes)
        target_data = build_target(output_data, gt_data, h, w)
        target_variable = [v for v in target_data]

        # for target in target_data:
        #     print(target.sum())
        box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

        
        loss = box_loss + iou_loss + class_loss
    
        out = scores
        loss.backward()
        loss_grads = out.grad
        # loss_grads = dout


        # loss_grad = elementwise_grad(lambda out: yolo_loss((output_data - target_variable) ** 2))(out)
        # loss_grads = grad_loss_wrt_out(out)

        # loss_grads = loss_grads.reshape(scores.shape)
        # loss

        # grads = compute_gradients(yolo_loss, output_variable, target_variable)


        # grad_loss_wrt_out = elementwise_grad(compute_loss, argnum = 0)

        # loss_grads = grad_loss_wrt_out(scores, target_variable)

        # print(grads[0].shape, grads[1].shape, grads[2].shape)
        # print(f"\nLoss = {loss}")
        # out = scores

        # loss = yolo_loss(output_variable, target_variable)
        # out.retain_grad()
        # loss.backward()
        # dout = out.grad.detach()

        # loss_grads = np.concatenate([grads[0], grads[1]], axis=-1)
        # loss_grads = np.concatenate([loss_grads, grads[2]], axis=-1)
        # loss_grads = loss_grads.reshape(scores.shape)

        # for grad in grads:
            #  print(grad.shape)

        # dout = out.grad()
        # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
        # dout = pickle.load(dout)
        # print('\n\n',dout.dtype, dout[dout!=0])
        return loss, loss_grads
    
    def backward(self, dout, cache):
        grads={}
        dout = to_torch(dout)
        dout = dout.cuda()
        last_dout, dw, db  = Torch_FastConvWB.backward(dout, cache['8'])
        grads['conv9.0.weight'], grads['conv9.0.bias'] = dw, db

        last_dout, grads['conv8.weight'], grads['bn8.weight'], grads['bn8.bias']  = Torch_Conv_BatchNorm_ReLU.backward      (last_dout, cache['7'])
        last_dout, grads['conv7.weight'], grads['bn7.weight'], grads['bn7.bias']  = Torch_Conv_BatchNorm_ReLU.backward      (last_dout, cache['6'])
        last_dout, grads['conv6.weight'], grads['bn6.weight'], grads['bn6.bias']  = Torch_Conv_BatchNorm_ReLU.backward      (last_dout, cache['5'])
        last_dout, grads['conv5.weight'], grads['bn5.weight'], grads['bn5.bias']  = Torch_Conv_BatchNorm_ReLU_Pool.backward (last_dout, cache['4'])
        last_dout, grads['conv4.weight'], grads['bn4.weight'], grads['bn4.bias']  = Torch_Conv_BatchNorm_ReLU_Pool.backward (last_dout, cache['3'])
        last_dout, grads['conv3.weight'], grads['bn3.weight'], grads['bn3.bias']  = Torch_Conv_BatchNorm_ReLU_Pool.backward (last_dout, cache['2'])
        last_dout, grads['conv2.weight'], grads['bn2.weight'], grads['bn2.bias']  = Torch_Conv_BatchNorm_ReLU_Pool.backward (last_dout, cache['1'])
        last_dout, grads['conv1.weight'], grads['bn1.weight'], grads['bn1.bias']  = Torch_Conv_BatchNorm_ReLU_Pool.backward (last_dout, cache['0'])
        # print(f"\n\t grads['W0']\n\t\t{grads['W0'].shape}\n\t\t{grads['W0'][grads['W0']!=0]}\n")
        return last_dout, grads

################################################################################
################################################################################
###############################  Functions Used  ###############################
################################################################################
################################################################################
    
class last_layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv9 = nn.Conv2d(1024, 125, kernel_size=1, stride=1, padding=0, bias=True)

    def Forward(self, x):
        
        return self.conv9(x)

def build_target(output, gt_data, H, W):
        """
        Build the training target for output tensor

        Arguments:

        output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
        gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

        delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
        conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
        class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)

        gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values
                                            (x1, y1, x2, y2) range 0~1
        gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
        num_obj_batch -- tensor of shape (B, 1). number of objects


        Returns:
        iou_target -- tensor of shape (B, H * W * num_anchors, 1)
        iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
        box_target -- tensor of shape (B, H * W * num_anchors, 4)
        box_mask -- tensor of shape (B, H * W * num_anchors, 1)
        class_target -- tensor of shape (B, H * W * num_anchors, 1)
        class_mask -- tensor of shape (B, H * W * num_anchors, 1)

        """
        delta_pred_batch = output[0]
        conf_pred_batch = output[1]
        class_score_batch = output[2]

        gt_boxes_batch = gt_data[0]
        gt_classes_batch = gt_data[1]
        num_boxes_batch = gt_data[2]

        bsize = delta_pred_batch.shape[0]



        num_anchors = 5  # hard code for now

        # initial the output tensor
        # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
        # what tensor is used doesn't matter

        # Convert memoryview to NumPy array
        delta_pred_batch = np.array(delta_pred_batch)

        iou_target = np.zeros((bsize, H * W, num_anchors, 1))
        iou_mask = np.ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale


        box_target = np.zeros((bsize, H * W, num_anchors, 4))
        box_mask = np.zeros((bsize, H * W, num_anchors, 1))

        class_target = np.zeros((bsize, H * W, num_anchors, 1))
        class_mask = np.zeros((bsize, H * W, num_anchors, 1))

        # iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
        # iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

        # box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
        # box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

        # class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
        # class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

        # get all the anchors

        anchors = np.array(cfg.anchors, dtype=np.float32)
        # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
        # this is very crucial because the predict output is normalized to 0~1, which is also
        # normalized by the grid width and height
        all_grid_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
        all_grid_xywh = np.array(all_grid_xywh, copy=True)

        # Create a copy of all_grid_xywh
        all_anchors_xywh = np.array(all_grid_xywh, copy=True)

        # Set the first two columns to 0.5
        all_anchors_xywh[:, 0:2] += 0.5

        if cfg.debug:
            print('all grid: ', all_grid_xywh[:12, :])
            print('all anchor: ', all_anchors_xywh[:12, :])
        # all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
        # all_anchors_xywh = all_grid_xywh.clone()
        # all_anchors_xywh[:, 0:2] += 0.5
        # if cfg.debug:
        #         print('all grid: ', all_grid_xywh[:12, :])
        #         print('all anchor: ', all_anchors_xywh[:12, :])
        all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)


        # process over batches
        for b in range(bsize):
                num_obj = num_boxes_batch[b].item()
                delta_pred = delta_pred_batch[b]
                gt_boxes = gt_boxes_batch[b][:num_obj, :]
                gt_classes = gt_classes_batch[b][:num_obj]

                # rescale ground truth boxes
                gt_boxes[:, 0::2] *= W
                gt_boxes[:, 1::2] *= H


                # step 1: process IoU target

                
                # apply delta_pred to pre-defined anchors
                all_anchors_xywh = all_anchors_xywh.reshape(-1, 4)
                box_pred = box_transform_inv(all_grid_xywh, delta_pred)
                box_pred = xywh2xxyy(box_pred)

                # for each anchor, its iou target is corresponded to the max iou with any gt boxes
                ious = box_ious(box_pred, gt_boxes) # shape: (H * W * num_anchors, num_obj)
                ious = ious.reshape(-1, num_anchors, num_obj)
                max_iou = np.max(ious, axis=-1, keepdims=True)  # shape: (H * W, num_anchors, 1)
                # max_iou, _ = torch.max(ious, dim=-1, keepdim=True) # shape: (H * W, num_anchors, 1)
                if cfg.debug:
                        print('ious', ious)

                # iou_target[b] = max_iou

                # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
                iou_thresh_filter = max_iou.reshape(-1) > cfg.thresh
                # n_pos = torch.nonzero(iou_thresh_filter).numel()
                n_pos = np.count_nonzero(iou_thresh_filter)
                if n_pos > 0:
                        iou_mask[b][max_iou >= cfg.thresh] = 0

                # step 2: process box target and class target
                # calculate overlaps between anchors and gt boxes
                overlaps = box_ious(all_anchors_xxyy, gt_boxes).reshape(-1, num_anchors, num_obj)
                gt_boxes_xywh = xxyy2xywh(gt_boxes)

                # iterate over all objects

                for t in range(gt_boxes.shape[0]):
                        # compute the center of each gt box to determine which cell it falls on
                        # assign it to a specific anchor by choosing max IoU

                        # gt_box_xywh = gt_boxes_xywh[t]
                        # gt_class = gt_classes[t]
                        gt_box_xywh = gt_boxes_xywh[t].cpu().detach().numpy()
                        gt_class = gt_classes[t].cpu().detach().numpy()
                        cell_idx_x, cell_idx_y = np.floor(gt_box_xywh[:2])
                        # cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
                        cell_idx = cell_idx_y * W + cell_idx_x
                        # cell_idx = cell_idx.long()
                        cell_idx = cell_idx.astype(np.int64)

                        # update box_target, box_mask
                        overlaps_in_cell = overlaps[cell_idx, :, t]
                        argmax_anchor_idx = np.argmax(overlaps_in_cell)

                        assigned_grid = all_grid_xywh.reshape(-1, num_anchors, 4)[None, cell_idx, argmax_anchor_idx, :]
                        gt_box = gt_box_xywh[None, :]
                        target_t = box_transform(assigned_grid, gt_box)
                        if cfg.debug:
                                print('assigned_grid, ', assigned_grid)
                                print('gt: ', gt_box)
                                print('target_t, ', target_t)
                        box_target[b, cell_idx, argmax_anchor_idx, :] = target_t[None, :]
                        box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

                        # update cls_target, cls_mask
                        class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
                        class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

                        # update iou target and iou mask
                        iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
                        if cfg.debug:
                                print(max_iou[cell_idx, argmax_anchor_idx, :])
                        iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

        return iou_target.reshape(bsize, -1, 1), \
                    iou_mask.reshape(bsize, -1, 1), \
                    box_target.reshape(bsize, -1, 4),\
                    box_mask.reshape(bsize, -1, 1), \
                    class_target.reshape(bsize, -1, 1).astype(np.int64), \
                    class_mask.reshape(bsize, -1, 1).astype(np.int64)

# def yolo_loss_midchange(output, target):
# 		"""
# 		Build yolo loss

# 		Arguments:
# 		output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
# 		target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

# 		delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
# 		conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
# 		class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

# 		iou_target -- Variable of shape (B, H * W * num_anchors, 1)
# 		iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
# 		box_target -- Variable of shape (B, H * W * num_anchors, 4)
# 		box_mask -- Variable of shape (B, H * W * num_anchors, 1)
# 		class_target -- Variable of shape (B, H * W * num_anchors, 1)
# 		class_mask -- Variable of shape (B, H * W * num_anchors, 1)

# 		Return:
# 		loss -- yolo overall multi-task loss
# 		"""

# 		delta_pred_batch = output[0]
# 		conf_pred_batch = output[1]
# 		class_score_batch = output[2]

# 		iou_target = target[0]
# 		iou_mask = target[1]
# 		box_target = target[2]
# 		box_mask = target[3]
# 		class_target = target[4]
# 		class_mask = target[5]


# 		b, _, num_classes = class_score_batch.shape
# 		class_score_batch = class_score_batch.reshape(-1, num_classes)
# 		class_target = class_target.reshape(-1)
# 		class_mask = class_mask.reshape(-1)



# 		# ignore the gradient of noobject's target
# 		# class_keep = to_torch(class_mask).nonzero().squeeze(1)
        
# 		class_keep = np.nonzero(class_mask)[0]

# 		class_score_batch_keep = class_score_batch[class_keep, :]
# 		class_target_keep = class_target[class_keep]


# 		# if cfg.debug:
# 		#     print(class_score_batch_keep)
# 		#     print(class_target_keep)
# 		# calculate the loss, normalized by batch size.
# 		# delta_pred_batch = to_torch(delta_pred_batch)
# 		# box_mask = to_torch(box_mask)
# 		# box_target = to_torch(box_target)


# 		box_loss = 1 / b * cfg.coord_scale * mse_loss_numpy(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0

# 		# conf_pred_batch = to_torch(conf_pred_batch)
# 		# iou_mask = to_torch(iou_mask)
# 		# iou_target = to_torch(iou_target)
# 		iou_loss = 1 / b * mse_loss_numpy(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0

# 		class_score_batch_keep = to_torch(class_score_batch_keep)
# 		class_target_keep = to_torch(class_target_keep).long()
# 		class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

# 		return box_loss, iou_loss, class_loss

def yolo_loss(output, target):
        """
        Build yolo loss

        Arguments:
        output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
        target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

        delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
        conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
        class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

        iou_target -- Variable of shape (B, H * W * num_anchors, 1)
        iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
        box_target -- Variable of shape (B, H * W * num_anchors, 4)
        box_mask -- Variable of shape (B, H * W * num_anchors, 1)
        class_target -- Variable of shape (B, H * W * num_anchors, 1)
        class_mask -- Variable of shape (B, H * W * num_anchors, 1)

        Return:
        loss -- yolo overall multi-task loss
        """

        delta_pred_batch = output[0]
        conf_pred_batch = output[1]
        class_score_batch = output[2]

        iou_target = target[0]
        iou_mask = target[1]
        box_target = target[2]
        box_mask = target[3]
        class_target = target[4]
        class_mask = target[5]


        b, _, num_classes = class_score_batch.shape
        class_score_batch = class_score_batch.reshape(-1, num_classes)
        class_target = class_target.reshape(-1)
        class_mask = class_mask.reshape(-1)



        # ignore the gradient of noobject's target
        class_keep = np.nonzero(class_mask)[0]
        # class_keep = to_torch(np.nonzero(class_mask)[0])

        class_score_batch_keep = class_score_batch[class_keep, :]
        class_target_keep = class_target[class_keep]


        # if cfg.debug:
        #     print(class_score_batch_keep)
        #     print(class_target_keep)
        # calculate the loss, normalized by batch size.
        # delta_pred_batch = to_torch(delta_pred_batch)
        # box_mask = to_torch(box_mask)
        # box_target = to_torch(box_target)


        box_loss = 1 / b * cfg.coord_scale * mse_loss_numpy(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0

        # conf_pred_batch = to_torch(conf_pred_batch)
        # iou_mask = to_torch(iou_mask)
        # iou_target = to_torch(iou_target)
        iou_loss = 1 / b * mse_loss_numpy(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0

        # class_score_batch_keep = to_torch(class_score_batch_keep)
        # class_target_keep = to_torch(class_target_keep).long()
        # print(class_score_batch_keep, class_target_keep)
        class_loss = 1 / b * cfg.class_scale * cross_entropy_numpy(class_score_batch_keep, class_target_keep)
        
        total_loss = box_loss + iou_loss + class_loss

        return box_loss, iou_loss, class_loss

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                                                dtype=torch.float64):
    """
    Implement Kaiming initialization for linear and convolution layers.
    
    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions for
        this layer
    - K: If K is None, then initialize weights for a linear layer with Din input
        dimensions and Dout output dimensions. Otherwise if K is a nonnegative
        integer then initialize the weights for a convolution layer with Din input
        channels, Dout output channels, and a kernel size of KxK.
    - relu: If Torch_ReLU=True, then initialize weights with a gain of 2 to account for
        a Torch_ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
        with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer. For a
        linear layer it should have shape (Din, Dout); for a convolution layer it
        should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:

        weight_scale = gain/(Din)
        weight = torch.zeros(Din,Dout, dtype=dtype,device = device)
        weight += weight_scale*torch.randn(Din,Dout, dtype=dtype,device= device)
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    else:

        weight_scale = gain/(Din*K*K)
        weight = torch.zeros(Din,Dout, K,K, dtype=dtype,device = device)
        weight += weight_scale*torch.randn(Din,Dout, K,K, dtype=dtype,device= device)
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    return weight

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[torch.arange(N), y]
    margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)
    margins[torch.arange(N), y] = 0.
    loss = margins.sum() / N
    num_pos = (margins > 0).sum(dim=1)
    dx = torch.zeros_like(x)
    dx[margins > 0] = 1.
    dx[torch.arange(N), y] -= num_pos.to(dx.dtype)
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0/ N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx

def box_ious(box1, box2):
        """
        Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

        Arguments:
        box1 -- tensor of shape (N, 4), first set of boxes
        box2 -- tensor of shape (K, 4), second set of boxes

        Returns:
        ious -- tensor of shape (N, K), ious between boxes
        """

        N = box1.shape[0]
        K = box2.shape[0]

        box2 = np.array(box2.cpu().detach())

        # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
        # xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
        # yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
        # xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
        # yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))
        xi1 = np.maximum(box1[:, 0].reshape(N, 1), box2[:, 0].reshape(1, K))
        yi1 = np.maximum(box1[:, 1].reshape(N, 1), box2[:, 1].reshape(1, K))
        xi2 = np.minimum(box1[:, 2].reshape(N, 1), box2[:, 2].reshape(1, K))
        yi2 = np.minimum(box1[:, 3].reshape(N, 1), box2[:, 3].reshape(1, K))



        # we want to compare the compare the value with 0 elementwise. However, we can't
        # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
        # what we want.
        # To feed a tensor 0 of same type and device with box1 and box2
        # we use tensor.new().fill_(0)

        # iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
        # ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))
        iw = np.maximum(xi2 - xi1, 0)
        ih = np.maximum(yi2 - yi1, 0)

        inter = iw * ih

        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        box1_area = box1_area[:, None]
        box2_area = box2_area[None, :]

        union_area = box1_area + box2_area - inter

        ious = inter / union_area

        return ious

def xxyy2xywh(box):
        """
        Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

        Arguments:
        box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

        Returns:
        xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
        """

        c_x = (box[:, 2] + box[:, 0]) / 2
        c_y = (box[:, 3] + box[:, 1]) / 2
        w = box[:, 2] - box[:, 0]
        h = box[:, 3] - box[:, 1]

        c_x = c_x.view(-1, 1)
        c_y = c_y.view(-1, 1)
        w = w.view(-1, 1)
        h = h.view(-1, 1)

        xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
        return xywh_box

def xywh2xxyy(box):
        """
        Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

        Arguments:
        box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

        Returns:
        xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
        """
        x1 = box[:, 0] - (box[:, 2]) / 2
        y1 = box[:, 1] - (box[:, 3]) / 2
        x2 = box[:, 0] + (box[:, 2]) / 2
        y2 = box[:, 1] + (box[:, 3]) / 2

        x1 = x1[:, None]
        y1 = y1[:, None]
        x2 = x2[:, None]
        y2 = y2[:, None]
        # y1 = y1.view(-1, 1)
        # x2 = x2.view(-1, 1)
        # y2 = y2.view(-1, 1)

        xxyy_box = np.concatenate([x1, y1, x2, y2], axis=1)
        return xxyy_box

def box_transform(box1, box2):
        """
        Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to box2

        Arguments:
        box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
        box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

        Returns:
        deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                                    used for transforming boxes to reference boxes
        """

        t_x = box2[:, 0] - box1[:, 0]
        t_y = box2[:, 1] - box1[:, 1]
        t_w = box2[:, 2] / box1[:, 2]
        t_h = box2[:, 3] / box1[:, 3]

        t_x = t_x.reshape(-1, 1)
        t_y = t_y.reshape(-1, 1)
        t_w = t_w.reshape(-1, 1)
        t_h = t_h.reshape(-1, 1)

        # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
        deltas = np.concatenate([t_x, t_y, t_w, t_h], axis=1)
        return deltas

def box_transform_inv(box, deltas):
        """
        apply deltas to box to generate predicted boxes

        Arguments:
        box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
        deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

        Returns:
        pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
        """

        c_x = box[:, 0] + deltas[:, 0]
        c_y = box[:, 1] + deltas[:, 1]
        w = box[:, 2] * deltas[:, 2]
        h = box[:, 3] * deltas[:, 3]

        c_x = c_x[:, None]
        c_y = c_y[:, None]
        w = w[:, None]
        h = h[:, None]

        pred_box = np.concatenate([c_x, c_y, w, h], axis=-1)
        return pred_box

# def generate_all_anchors(anchors, H, W):
#         """
#         Generate dense anchors given grid defined by (H,W)

#         Arguments:
#         anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
#         H -- int, grid height
#         W -- int, grid width

#         Returns:
#         all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
#         """

#         # number of anchors per cell
#         A = anchors.size(0)

#         # number of cells
#         K = H * W

#         shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

#         # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
#         shift_x = shift_x.t().contiguous()
#         shift_y = shift_y.t().contiguous()

#         # shift_x is a long tensor, c_x is a float tensor
#         c_x = shift_x.float()
#         c_y = shift_y.float()

#         centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

#         # add anchors width and height to centers
#         all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
#                                                         anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

#         all_anchors = all_anchors.view(-1, 4)

#         return all_anchors

def generate_all_anchors(anchors, H, W):
    """
    Generate dense anchors given grid defined by (H,W)

    Arguments:
    anchors -- ndarray of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- ndarray of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.shape[0]

    # number of cells
    K = H * W

    shift_x, shift_y = np.meshgrid(np.arange(W), np.arange(H))

    # flatten and transpose shift_x and shift_y
    shift_x = shift_x.ravel()
    shift_y = shift_y.ravel()

    # shift_x is an integer array, c_x is a float array
    c_x = shift_x.astype(np.float32)
    c_y = shift_y.astype(np.float32)

    centers = np.stack([c_x, c_y], axis=-1)  # array of shape (H * W, 2), (cx, cy)

    # add anchors width and height to centers
    centers_expanded = np.expand_dims(centers, axis=1).repeat(A, axis=1)  # shape: (K, A, 2)
    anchors_expanded = np.expand_dims(anchors, axis=0).repeat(K, axis=0)  # shape: (K, A, 2)

    all_anchors = np.concatenate([centers_expanded, anchors_expanded], axis=-1)

    all_anchors = all_anchors.reshape(-1, 4)

    return all_anchors



################################################################################
################################################################################
#################   Pytorch Implementations and Sandwich Layers  ###############
################################################################################
################################################################################


class Torch_Conv(object):

    @staticmethod
    def Forward(x, w, b, conv_param):
        """
        A naive implementation of the Forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
            - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
            - 'pad': The number of pixels that will be used to zero-pad the input. 
            
        During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
        along the height and width axes of the input. Be careful not to modfiy the original
        input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None

        pad = conv_param['pad']
        stride = conv_param['stride']
        N,C,H,W = x.shape
        F,C,HH,WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        x = torch.nn.functional.pad(x, (pad,pad,pad,pad))
        
        out = torch.zeros((N,F,H_out,W_out),dtype =  x.dtype, device = x.device)

        for n in range(N):
            for f in range(F):
                for height in range(H_out):
                    for width in range(W_out):
                        out[n,f,height,width] = (x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] *w[f]).sum() + b[f]

        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - z: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_Forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None

        x, w, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        N,F,H_dout,W_dout = dout.shape
        F,C,HH,WW = w.shape
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        dw[f]+= x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] * dout[n,f,height,width]
                        dx[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW]+=w[f] * dout[n,f,height,width]
                
        dx = dx[:,:,1:-1,1:-1] #delete padded "pixels"

        return dx, dw


class Torch_ConvB(object):

    @staticmethod
    def Forward(x, w, b, conv_param):
        """
        A naive implementation of the Forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
            - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
            - 'pad': The number of pixels that will be used to zero-pad the input. 
            
        During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
        along the height and width axes of the input. Be careful not to modfiy the original
        input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None

        pad = conv_param['pad']
        stride = conv_param['stride']
        N,C,H,W = x.shape
        F,C,HH,WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        x = torch.nn.functional.pad(x, (pad,pad,pad,pad))
        
        out = torch.zeros((N,F,H_out,W_out),dtype =  x.dtype, device = x.device)

        for n in range(N):
            for f in range(F):
                for height in range(H_out):
                    for width in range(W_out):
                        out[n,f,height,width] = (x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] *w[f]).sum() + b[f]

        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_Forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None

        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        N,F,H_dout,W_dout = dout.shape
        F,C,HH,WW = w.shape
        db = torch.zeros_like(b)
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        db[f]+=dout[n,f,height,width]
                        dw[f]+= x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] * dout[n,f,height,width]
                        dx[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW]+=w[f] * dout[n,f,height,width]
                
        dx = dx[:,:,1:-1,1:-1] #delete padded "pixels"

        return dx, dw, db

class Torch_MaxPool(object):

    @staticmethod
    def Forward(x, pool_param):
        """
        A naive implementation of the Forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
            - 'pool_height': The height of each pooling region
            - 'pool_width': The width of each pooling region
            - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output data, of shape (N, C, H', W') where H' and W' are given by
            H' = 1 + (H - pool_height) / stride
            W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None

        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        N,C,H,W = x.shape
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        out = torch.zeros((N,C,H_out,W_out),dtype =  x.dtype, device = x.device)
        for n in range(N):
                for height in range(H_out):
                    for width in range(W_out):
                        val, _ = x[n,:,height*stride:height*stride+pool_height,width*stride:width*stride+pool_width].reshape(C,-1).max(dim = 1)
                        out[n,:,height,width] = val

        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the Forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None

        x, pool_param = cache
        N,C,H,W = x.shape
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        dx = torch.zeros_like(x)
        for n in range(N):
                for height in range(H_out):
                    for width in range(W_out):
                        local_x  = x[n,:,height*stride:height*stride+pool_height,width*stride:width*stride+pool_width]
                        shape_local_x = local_x.shape
                        reshaped_local_x = local_x.reshape(C ,-1)
                        local_dw = torch.zeros_like(reshaped_local_x)
                        values, indicies = reshaped_local_x.max(-1)
                        local_dw[range(C),indicies] =  dout[n,:,height,width]
                        dx[n,:,height*stride:height*stride+pool_height,width*stride:width*stride+pool_width] = local_dw.reshape(shape_local_x)

        return dx

class Torch_BatchNorm(object):

    @staticmethod
    def Forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance are
        computed from minibatch statistics and used to normalize the incoming data.
        During training we also keep an exponentially decaying running mean of the
        mean and variance of each feature, and these averages are used to normalize
        data at test-time.

        At each timestep we update the running averages for mean and variance using
        an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using a
        large number of training images rather than using a running average. For
        this implementation we have chosen to use running averages instead since
        they do not require an additional estimation step; the PyTorch
        implementation of batch normalization also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - momentum: Constant for running mean / variance.
            - running_mean: Array of shape (D,) giving running mean of features
            - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

        out, cache = None, None
        if mode == 'train':
            #######################################################################
            # TO DO: Implement the training-time Forward pass for batch norm.      #
            # Use minibatch statistics to compute the mean and variance, use      #
            # these statistics to normalize the incoming data, and scale and      #
            # shift the normalized data using gamma and beta.                     #
            #                                                                     #
            # You should store the output in the variable out. Any intermediates  #
            # that you need for the backward pass should be stored in the cache   #
            # variable.                                                           #
            #                                                                     #
            # You should also use your computed sample mean and variance together #
            # with the momentum variable to update the running mean and running   #
            # variance, storing your result in the running_mean and running_var   #
            # variables.                                                          #
            #                                                                     #
            # Note that though you should be keeping track of the running         #
            # variance, you should normalize the data based on the standard       #
            # deviation (square root of variance) instead!                        # 
            # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
            # might prove to be helpful.                                          #
            #######################################################################
            # Replace "pass" statement with your code
            #step1: calculate mean
            running_mean = running_mean.to(x.device)
            running_var = running_var.to(x.device)
            mu = 1./N * torch.sum(x, axis = 0)
            running_mean = momentum * running_mean + (1 - momentum) * mu

            #step2: subtract mean vector of every trainings example
            xmu = x - mu
            
            #step3: following the lower branch - calculation denominator
            sq = xmu ** 2
            
            #step4: calculate variance
            var = 1./N * torch.sum(sq, axis = 0)
            running_var = momentum * running_var + (1 - momentum) * var
            #step5: add eps for numerical stability, then sqrt
            sqrtvar = torch.sqrt(var + eps)

            #step6: invert sqrtwar
            ivar = 1./sqrtvar
        
            #step7: execute normalization
            xhat = xmu * ivar

            #step8: Nor the two transformation steps
            #print(gamma)

            gammax = gamma * xhat

            #step9
            out = gammax + beta

            cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        elif mode == 'test':
            #######################################################################
            # TO DO: Implement the test-time Forward pass for batch normalization. #
            # Use the running mean and variance to normalize the incoming data,   #
            # then scale and shift the normalized data using gamma and beta.      #
            # Store the result in the out variable.                               #
            #######################################################################
            # Replace "pass" statement with your code
            normolized = ((x - running_mean)/(running_var + eps)**(1/2))
            out = normolized * gamma + beta
            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        else:
            raise ValueError('Invalid Forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a computation graph for
        batch normalization on paper and propagate gradients backward through
        intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_Forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        ###########################################################################
        # TO DO: Implement the backward pass for batch normalization. Store the    #
        # results in the dx, dgamma, and dbeta variables.                         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
        # might prove to be helpful.                                              #
        # Don't forget to implement train and test mode separately.               #
        ###########################################################################
        # Replace "pass" statement with your code
        xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
        
        N,D = dout.shape

        #step9
        dbeta = torch.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable

        #step8
        dgamma = torch.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma

        #step7
        divar = torch.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        #step6
        dsqrtvar = -1. /(sqrtvar**2) * divar

        #step5
        dvar = 0.5 * 1. / torch.sqrt(var+eps) * dsqrtvar

        #step4
        dsq = 1. /N * torch.ones((N,D),device = dout.device) * dvar

        #step3
        dxmu2 = 2 * xmu * dsq

        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * torch.sum(dxmu1+dxmu2, axis=0)

        #step1
        dx2 = 1. /N * torch.ones((N,D),device = dout.device) * dmu

        #step0
        dx = dx1 + dx2
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives for the batch
        normalizaton backward pass on paper and simplify as much as possible. You
        should be able to derive a simple expression for the backward pass. 
        See the jupyter notebook for more hints.
        
        Note: This implementation should expect to receive the same cache variable
        as batchnorm_backward, but might not use all of the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###########################################################################
        # TO DO: Implement the backward pass for batch normalization. Store the    #
        # results in the dx, dgamma, and dbeta variables.                         #
        #                                                                         #
        # After computing the gradient with respect to the centered inputs, you   #
        # should be able to compute gradients with respect to the inputs in a     #
        # single statement; our implementation fits on a single 80-character line.#
        ###########################################################################
        # Replace "pass" statement with your code
        xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
        N,D = dout.shape
        # get the dimensions of the input/output
        dbeta = torch.sum(dout, dim=0)
        dgamma = torch.sum(xhat * dout, dim=0)
        dx = (gamma*ivar/N) * (N*dout - xhat*dgamma - dbeta)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return dx, dgamma, dbeta

class Torch_SpatialBatchNorm(object):

    @staticmethod
    def Forward(x, gamma, beta, bn_param):
        """
        Computes the Forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - momentum: Constant for running mean / variance. momentum=0 means that
            old information is discarded completely at every time step, while
            momentum=1 means that new information is never incorporated. The
            default of momentum=0.9 should work well in most situations.
            - running_mean: Array of shape (C,) giving running mean of features
            - running_var Array of shape (C,) giving running variance of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ###########################################################################
        # TO DO: Implement the Forward pass for spatial batch normalization.       #
        #                                                                         #
        # HINT: You can implement spatial batch normalization by calling the      #
        # vanilla version of batch normalization you implemented above.           #
        # Your implementation should be very short; ours is less than five lines. #
        ###########################################################################
        # Replace "pass" statement with your code
        N,C,H,W = x.shape
        pre_m = x.permute(1,0,2,3).reshape(C,-1).T
        pre_m_normolized, cache= Torch_BatchNorm.Forward(pre_m, gamma, beta, bn_param)
        out = pre_m_normolized.T.reshape(C, N, H, W).permute(1,0,2,3)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the Forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        ###########################################################################
        # TO DO: Implement the backward pass for spatial batch normalization.      #
        #                                                                         #
        # HINT: You can implement spatial batch normalization by calling the      #
        # vanilla version of batch normalization you implemented above.           #
        # Your implementation should be very short; ours is less than five lines. #
        ###########################################################################
        # Replace "pass" statement with your code
        N,C,H,W = dout.shape
        pre_m = dout.permute(1,0,2,3).reshape(C,-1).T
        dx, dgamma, dbeta = Torch_BatchNorm.backward_alt(pre_m, cache)
        dx =dx.T.reshape(C, N, H, W).permute(1,0,2,3)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return dx, dgamma, dbeta

class Torch_FastConv(object):

    @staticmethod
    def Forward(x, w, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad, bias=False)
        layer.weight = torch.nn.Parameter(w)
        # layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            # db = layer.bias.grad.detach()
            layer.weight.grad  = None
        except RuntimeError:
            dx, dw = torch.zeros_like(tx), torch.zeros_like(layer.weight)
        return dx, dw

class Torch_FastConvWB(object):

    @staticmethod
    def Forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        # try:
        x, _, _, _, tx, out, layer = cache
        out.backward(dout)
        dx = tx.grad.detach()
        dw = layer.weight.grad.detach()
        db = layer.bias.grad.detach()
        layer.weight.grad = layer.bias.grad = None
        # except RuntimeError:
        #   dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
        return dx, dw, db

class Torch_FastMaxPool(object):

    @staticmethod
    def Forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx

class Torch_Padd(object):

    @staticmethod
    def Forward(x, pad_param):
        layer = torch.nn.ReflectionPad2d(pad_param)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pad_param, tx, out, layer)

        return out, cache
    
    @staticmethod
    def backward(dout, cache):
        x, _, tx, out, layer = cache
        out.backward(dout)
        dx = tx.grad.detach()
        return dx

class Torch_Conv_ReLU(object):

    @staticmethod
    def Forward(x, w, conv_param):
        """
        A convenience layer that performs a convolution followed by a Torch_ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the convolutional layer
        Returns a tuple of:
        - out: Output from the Torch_ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = Torch_FastConv.Forward(x, w, conv_param)
        out, relu_cache = Torch_ReLU.Forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = Torch_ReLU.backward(dout, relu_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw

class Torch_Conv_ReLU_Pool(object):

    @staticmethod
    def Forward(x, w, conv_param, pool_param):
        """
        A convenience layer that performs a convolution, a Torch_ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = Torch_FastConv.Forward(x, w, conv_param)
        s, relu_cache = Torch_ReLU.Forward(a)
        out, pool_cache = Torch_FastMaxPool.Forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = Torch_FastMaxPool.backward(dout, pool_cache)
        da = Torch_ReLU.backward(ds, relu_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw

class Torch_Conv_BatchNorm_ReLU(object):

    @staticmethod
    def Forward(x, w, gamma, beta, conv_param, bn_param):
        a, conv_cache = Torch_FastConv.Forward(x, w, conv_param)
        an, bn_cache = Torch_SpatialBatchNorm.Forward(a, gamma, beta, bn_param)
        out, relu_cache = Torch_ReLU.Forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = Torch_ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = Torch_SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw, dgamma, dbeta

class Torch_Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def Forward(x, w, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = Torch_FastConv.Forward(x, w, conv_param)
        an, bn_cache = Torch_SpatialBatchNorm.Forward(a, gamma, beta, bn_param)
        s, relu_cache = Torch_ReLU.Forward(an)
        out, pool_cache = Torch_FastMaxPool.Forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = Torch_FastMaxPool.backward(dout, pool_cache)
        dan = Torch_ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = Torch_SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw, dgamma, dbeta

class Torch_ReLU(object):

        @staticmethod
        def Forward(x, alpha=0.1):

                out = None
                out = x.clone()
                out[out < 0] = out[out < 0] * alpha
                cache = x

                return out, cache

        @staticmethod
        def backward(dout, cache, alpha=0.1):

                dx, x = None, cache

                dl = torch.ones_like(x)
                dl[x < 0] = alpha
                dx = dout * dl

                return dx
        


class Torch_FastMaxPool(object):

    @staticmethod
    def Forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx
        
class Torch_Pad2d(object):

    @staticmethod
    def Forward(x, pad_param):
        N, C, H, W = x.shape
        layer = torch.nn.ReflecionPad2d(pad_param)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pad_param, tx, out, layer)
        return out, cache
    
    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)

class Torch_Pad2d(object):

    @staticmethod
    def Forward(x, pad_param):
        N, C, H, W = x.shape
        layer = nn.ZeroPad2d(pad_param)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pad_param, tx, out, layer)
        return out, cache
    
    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx