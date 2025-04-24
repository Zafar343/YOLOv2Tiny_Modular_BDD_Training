import os
import torch
from copy import deepcopy as dc
import pickle
from models_torch.torch_original import *



def Save_File(_path, data):
    _dir = _path.split('/')[1:-1]
    if len(_dir)>1: _dir = os.path.join(_dir)
    else: _dir = _dir[0]
    if not os.path.isdir(_dir): os.mkdir(_dir)
    
    with open(_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
class Pytorch_bn(object):
    
    def __init__(self,num_classes=20):
        # self.self           = parent
        self.num_classes    = num_classes
        self.model          = None
        self.loss           = None
        self.optimizer      = None
        self.scheduler      = None
        self.device         = None
        self.train_loader   = None
        self.Weight = [[] for _ in range(9)]

        self.Bias = None
        self.Gamma = [[] for _ in range(8)]
        self.Beta = [[] for _ in range(8)]
        self.Running_Mean_Dec = [[] for _ in range(8)]
        self.Running_Var_Dec = [[] for _ in range(8)]
        self.gWeight = [[] for _ in range(9)]
        self.gBias = None
        self.gGamma = [[] for _ in range(8)]
        self.gBeta = [[] for _ in range(8)]


        # if parent != "none":
        #     self.Mode                 = self.self.Mode     
        #     self.Brain_Floating_Point = self.self.Brain_Floating_Point                     
        #     self.Exponent_Bits        = self.self.Exponent_Bits             
        #     self.Mantissa_Bits        = self.self.Mantissa_Bits   
        

        

        self.modtorch_model = DeepConvNetTorch(input_dims=(3, 416, 416),
                                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                        max_pools=[0, 1, 2, 3, 4],
                                        weight_scale='kaiming',
                                        num_classes=self.num_classes,
                                        batchnorm=True,
                                        dtype=torch.float32, device='cuda')
        
        self._device = self.modtorch_model.params['conv1.weight'].device


        self.optimizer_config = {}
        optim_config = {'learning_rate': 0.01, 'momentum': 0.9}
        for p, _ in self.modtorch_model.params.items():
            d = {k: v for k, v in optim_config.items()}
            self.optimizer_config[p] = d
            

    def get_grads(self):
        self.gWeight, self.gBias, self.gGamma, self.gBeta, self.gRunning_Mean_Dec, self.gRunning_Var_Dec = \
            dc(self.Weight), dc(self.Bias), dc(self.Gamma), dc(self.Beta), dc(self.Running_Mean_Dec), dc(self.Running_Var_Dec)
            
        self.gWeight[0]  = self.grads['conv1.weight']              
        self.gWeight[1]  = self.grads['conv2.weight']              
        self.gWeight[2]  = self.grads['conv3.weight']              
        self.gWeight[3]  = self.grads['conv4.weight']              
        self.gWeight[4]  = self.grads['conv5.weight']              
        self.gWeight[5]  = self.grads['conv6.weight']              
        self.gWeight[6]  = self.grads['conv7.weight']              
        self.gWeight[7]  = self.grads['conv8.weight']              
        self.gWeight[8]  = self.grads['conv9.0.weight']              
        self.gBias       = self.grads['conv9.0.bias']         
        self.gGamma[0]   = self.grads['bn1.weight']                       
        self.gGamma[1]   = self.grads['bn2.weight']                      
        self.gGamma[2]   = self.grads['bn3.weight']                      
        self.gGamma[3]   = self.grads['bn4.weight']                      
        self.gGamma[4]   = self.grads['bn5.weight']                      
        self.gGamma[5]   = self.grads['bn6.weight']                      
        self.gGamma[6]   = self.grads['bn7.weight']                      
        self.gGamma[7]   = self.grads['bn8.weight']                      
        self.gBeta[0]    = self.grads['bn1.bias']              
        self.gBeta[1]    = self.grads['bn2.bias']              
        self.gBeta[2]    = self.grads['bn3.bias']              
        self.gBeta[3]    = self.grads['bn4.bias']              
        self.gBeta[4]    = self.grads['bn5.bias']              
        self.gBeta[5]    = self.grads['bn6.bias']              
        self.gBeta[6]    = self.grads['bn7.bias']              
        self.gBeta[7]    = self.grads['bn8.bias']     
        
    def get_weights(self):
        self.Weight[0]              = self.modtorch_model.params['conv1.weight']              
        self.Weight[1]              = self.modtorch_model.params['conv2.weight']              
        self.Weight[2]              = self.modtorch_model.params['conv3.weight']              
        self.Weight[3]              = self.modtorch_model.params['conv4.weight']              
        self.Weight[4]              = self.modtorch_model.params['conv5.weight']              
        self.Weight[5]              = self.modtorch_model.params['conv6.weight']              
        self.Weight[6]              = self.modtorch_model.params['conv7.weight']              
        self.Weight[7]              = self.modtorch_model.params['conv8.weight']              
        self.Weight[8]              = self.modtorch_model.params['conv9.0.weight']              
        self.Bias                   = self.modtorch_model.params['conv9.0.bias']         
        self.Gamma[0]               = self.modtorch_model.params['bn1.weight']                      
        self.Gamma[1]               = self.modtorch_model.params['bn2.weight']                      
        self.Gamma[2]               = self.modtorch_model.params['bn3.weight']                      
        self.Gamma[3]               = self.modtorch_model.params['bn4.weight']                      
        self.Gamma[4]               = self.modtorch_model.params['bn5.weight']                      
        self.Gamma[5]               = self.modtorch_model.params['bn6.weight']                      
        self.Gamma[6]               = self.modtorch_model.params['bn7.weight']                      
        self.Gamma[7]               = self.modtorch_model.params['bn8.weight']                      
        self.Beta[0]                = self.modtorch_model.params['bn1.bias']              
        self.Beta[1]                = self.modtorch_model.params['bn2.bias']              
        self.Beta[2]                = self.modtorch_model.params['bn3.bias']              
        self.Beta[3]                = self.modtorch_model.params['bn4.bias']              
        self.Beta[4]                = self.modtorch_model.params['bn5.bias']              
        self.Beta[5]                = self.modtorch_model.params['bn6.bias']              
        self.Beta[6]                = self.modtorch_model.params['bn7.bias']              
        self.Beta[7]                = self.modtorch_model.params['bn8.bias']              
        self.Running_Mean_Dec[0]    = self.modtorch_model.params['bn1.running_mean']                        
        self.Running_Mean_Dec[1]    = self.modtorch_model.params['bn2.running_mean']                        
        self.Running_Mean_Dec[2]    = self.modtorch_model.params['bn3.running_mean']                        
        self.Running_Mean_Dec[3]    = self.modtorch_model.params['bn4.running_mean']                        
        self.Running_Mean_Dec[4]    = self.modtorch_model.params['bn5.running_mean']                        
        self.Running_Mean_Dec[5]    = self.modtorch_model.params['bn6.running_mean']                        
        self.Running_Mean_Dec[6]    = self.modtorch_model.params['bn7.running_mean']                        
        self.Running_Mean_Dec[7]    = self.modtorch_model.params['bn8.running_mean']                        
        self.Running_Var_Dec[0]     = self.modtorch_model.params['bn1.running_var']                       
        self.Running_Var_Dec[1]     = self.modtorch_model.params['bn2.running_var']                       
        self.Running_Var_Dec[2]     = self.modtorch_model.params['bn3.running_var']                       
        self.Running_Var_Dec[3]     = self.modtorch_model.params['bn4.running_var']                       
        self.Running_Var_Dec[4]     = self.modtorch_model.params['bn5.running_var']                       
        self.Running_Var_Dec[5]     = self.modtorch_model.params['bn6.running_var']                       
        self.Running_Var_Dec[6]     = self.modtorch_model.params['bn7.running_var']                       
        self.Running_Var_Dec[7]     = self.modtorch_model.params['bn8.running_var']                       
        
        
    def load_weights(self, data):
        try: self.Weight, self.Bias, self.Gamma, self.Beta, self.Running_Mean_Dec, self.Running_Var_Dec = data
        except: self.Weight, self.Bias, self.Gamma, self.Beta = data
        
        _device = self.modtorch_model.params['conv1.weight'].device

        self.modtorch_model.params['conv1.weight']            = self.Weight[0]
        self.modtorch_model.params['conv2.weight']            = self.Weight[1]
        self.modtorch_model.params['conv3.weight']            = self.Weight[2]
        self.modtorch_model.params['conv4.weight']            = self.Weight[3]
        self.modtorch_model.params['conv5.weight']            = self.Weight[4]
        self.modtorch_model.params['conv6.weight']            = self.Weight[5]
        self.modtorch_model.params['conv7.weight']            = self.Weight[6]
        self.modtorch_model.params['conv8.weight']            = self.Weight[7]
        self.modtorch_model.params['conv9.0.weight']            = self.Weight[8]
        self.modtorch_model.params['conv9.0.bias']            = self.Bias
        self.modtorch_model.params['bn1.weight']        = self.Gamma[0]
        self.modtorch_model.params['bn2.weight']        = self.Gamma[1]
        self.modtorch_model.params['bn3.weight']        = self.Gamma[2]
        self.modtorch_model.params['bn4.weight']        = self.Gamma[3]
        self.modtorch_model.params['bn5.weight']        = self.Gamma[4]
        self.modtorch_model.params['bn6.weight']        = self.Gamma[5]
        self.modtorch_model.params['bn7.weight']        = self.Gamma[6]
        self.modtorch_model.params['bn8.weight']        = self.Gamma[7]
        self.modtorch_model.params['bn1.bias']         = self.Beta[0]
        self.modtorch_model.params['bn2.bias']         = self.Beta[1]
        self.modtorch_model.params['bn3.bias']         = self.Beta[2]
        self.modtorch_model.params['bn4.bias']         = self.Beta[3]
        self.modtorch_model.params['bn5.bias']         = self.Beta[4]
        self.modtorch_model.params['bn6.bias']         = self.Beta[5]
        self.modtorch_model.params['bn7.bias']         = self.Beta[6]
        self.modtorch_model.params['bn8.bias']         = self.Beta[7]
        self.modtorch_model.params['bn1.running_mean'] = self.Running_Mean_Dec[0]
        self.modtorch_model.params['bn2.running_mean'] = self.Running_Mean_Dec[1]
        self.modtorch_model.params['bn3.running_mean'] = self.Running_Mean_Dec[2]
        self.modtorch_model.params['bn4.running_mean'] = self.Running_Mean_Dec[3]
        self.modtorch_model.params['bn5.running_mean'] = self.Running_Mean_Dec[4]
        self.modtorch_model.params['bn6.running_mean'] = self.Running_Mean_Dec[5]
        self.modtorch_model.params['bn7.running_mean'] = self.Running_Mean_Dec[6]
        self.modtorch_model.params['bn8.running_mean'] = self.Running_Mean_Dec[7]
        self.modtorch_model.params['bn1.running_var']  = self.Running_Var_Dec[0]
        self.modtorch_model.params['bn2.running_var']  = self.Running_Var_Dec[1]
        self.modtorch_model.params['bn3.running_var']  = self.Running_Var_Dec[2]
        self.modtorch_model.params['bn4.running_var']  = self.Running_Var_Dec[3]
        self.modtorch_model.params['bn5.running_var']  = self.Running_Var_Dec[4]
        self.modtorch_model.params['bn6.running_var']  = self.Running_Var_Dec[5]
        self.modtorch_model.params['bn7.running_var']  = self.Running_Var_Dec[6]
        self.modtorch_model.params['bn8.running_var']  = self.Running_Var_Dec[7]

        for nam, val in  self.modtorch_model.params.items():
            self.modtorch_model.params[nam] = val.to(_device)

    def Before_Forward(self, Input):
            pass
        
    def Forward(self, data):
        X = data.to(self._device)
        self.out, self.cache, self.Out_all_layers = self.modtorch_model.Forward(X)
        return self.out, self.cache, self.Out_all_layers

    def Forward_pred(self, out, num_anchors=5, num_classes=1):
        """
        Evaluate loss and gradient for the deep convolutional network.
        Input / output: Same API as ThreeLayerConvNet.
        """
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

        return delta_pred, conf_pred, class_pred
        
    def Backward(self, data):
        self.dout, self.grads = self.modtorch_model.backward(self.dout, self.cache)
        self.get_weights()
        self.get_grads()