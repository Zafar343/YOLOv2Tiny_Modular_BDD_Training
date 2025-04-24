import torch
import pickle
from models_torch.torch_original_compat import *
import time

def to_numpy(x):
	return x.cpu().detach().numpy()

def to_torch(x):
	return torch.from_numpy(x).float()

def sgd_momentum(w, dw, config=None):

  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)

  v = config.get('velocity', torch.zeros_like(w))

  next_w = None

  v = config['momentum']*v - config['learning_rate'] * dw
  next_w = w + v

  config['velocity'] = v

  return next_w, config

def sgd_momentum_update(Inputs=[], gInputs=[], epochs = 0, optimizer_config = None):
	weight, bias, gamma, beta = Inputs
	gweight, gbias, ggamma, gbeta = gInputs

	for i in range(8):
		gweight[i] = torch.clamp(gweight[i], -1, 1)
		gbias = torch.clamp(gbias, -1, 1)
		ggamma[i] = torch.clamp(ggamma[i], -1, 1)
		gbeta[i] = torch.clamp(gbeta[i], -1, 1)

	gweight[8] = torch.clamp(gweight[8], -1, 1)
	gbias = torch.clamp(gbias, -1, 1)

	#  Learning Rate
	# Initial LR = 0.01 gives NaN for LN
	# Initial LR = 0.001 --- gives best result when training from scratch
	# Initial LR = 0.0001 --- should be better for pre-trained results.
	# Initial LR = 0.00001 is probably too slow.

	# initial_lr = 0.0001 # initial learning rate
	# warmup_epochs = 10
	# plateau_epochs = 30
	# decay_rate = 0.1

	initial_lr = 0.0001  # Initial learning rate
	warmup_epochs = 5   # Number of epochs for warmup
	plateau_epochs = 30 # Number of epochs for plateau phase
	decay_rate = 0.98    # Decay rate

	if epochs < warmup_epochs:
		learning_rate = initial_lr * (epochs + 1) / warmup_epochs
	elif epochs < plateau_epochs:
		learning_rate = initial_lr
	else:
		learning_rate = initial_lr * (decay_rate ** (epochs - plateau_epochs))

	# i = 0

	# for i in range(8):
	# 	weight[i] = weight[i].cuda()
	# 	gamma[i] = gamma[i].cuda()
	# 	beta[i] = beta[i].cuda()
	# 	gweight[i] = gweight[i].cuda()
	# 	ggamma[i] = ggamma[i].cuda()
	# 	gbeta[i] = gbeta[i].cuda()
	# weight[8] = weight[8].cuda()
	# gweight[8] = gweight[8].cuda()
	# bias = bias.cuda()
	# gbias = gbias.cuda()

	config = {'learning_rate': learning_rate, 'momentum': 0.7}
	
	with torch.no_grad():
		for i in range(8):
			config = optimizer_config['W{}'.format(i)]
			config['learning_rate'] = learning_rate
			weight[i], next_config = sgd_momentum(weight[i], gweight[i], config)
			optimizer_config['W{}'.format(i)] = next_config

		# for i in range(8):
			config = optimizer_config['gamma{}'.format(i)]
			config['learning_rate'] = learning_rate
			gamma[i], next_config = sgd_momentum(gamma[i], ggamma[i].reshape(-1), config)
			optimizer_config['gamma{}'.format(i)] = next_config
		
		# for i in range(8):
			config = optimizer_config['beta{}'.format(i)]
			config['learning_rate'] = learning_rate
			beta[i], next_config = sgd_momentum(beta[i], gbeta[i].reshape(-1), config)
			optimizer_config['beta{}'.format(i)] = next_config

		config = optimizer_config['W8']
		config['learning_rate'] = learning_rate
		weight[8], next_config = sgd_momentum(weight[8], gweight[8], config)
		optimizer_config['W8'] = next_config

		config = optimizer_config['b8']
		config['learning_rate'] = learning_rate
		bias, next_config = sgd_momentum(bias, gbias, config)
		optimizer_config['b8'] = next_config
	
	return (weight, bias, gamma, beta), optimizer_config

with open("./Dataset/dummy_data.pkl", "rb") as f:
	dummy_data = pickle.load(f)

model = Pytorch_bn()
checkpoint = torch.load('./Dataset/yolov2_epoch_299.pth')
pytorch_model = checkpoint['model']

for param, val in model.modtorch_model.params.items():
	for param1, val1 in pytorch_model.items():
		if (param == param1):
			model.modtorch_model.params[param] = val1.cpu().detach()

with open("./Dataset/dummy_data.pkl", "rb") as f:
	dummy_data = pickle.load(f)

dummy_data = iter(dummy_data)
im_data, gt_boxes, gt_classes, num_obj = next(dummy_data)

print(f"Preprocess for initial weights and grads duration:")
model.out, model.cache, model.FOut = model.modtorch_model.Forward(im_data)
model.loss, model.loss_grad = model.modtorch_model.loss(model.out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)
model.dout, model.grads = model.modtorch_model.backward(model.loss_grad, model.cache)
model.get_weights()
model.get_grads()


# sample code
epochs = 10
f0 = time.time()
for epoch in range(epochs):

	with open("./Dataset/dummy_data.pkl", "rb") as f:
		dummy_data = pickle.load(f)

	dummy_data = iter(dummy_data)
	im_data, gt_boxes, gt_classes, num_obj = next(dummy_data)
	# im_data = im_data.cuda()
	# gt_boxes = gt_boxes.cuda()
	# gt_classes = gt_classes.cuda()
	# num_obj = num_obj.cuda()
	# f1 = time.time()
	print(f"Epoch {epoch}")
	model.out, model.cache, model.FOut = model.modtorch_model.Forward(im_data)
	# f2 = time.time()
	# print(f"Forward duration: {f2 - f1:.4f}")
	model.loss, model.loss_grad = model.modtorch_model.loss(model.out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)
	# f3 = time.time()
	# print(f"Calculate loss duration: {f3 - f2:.4f}")
	model.dout, model.grads = model.modtorch_model.backward(model.loss_grad, model.cache)
	# f4 = time.time()
	# print(f"Backward duration: {f4 - f3:.4f}")

	new_weights, optims = sgd_momentum_update(Inputs = [model.Weight,  model.Bias,  model.Gamma,  model.Beta],
									gInputs = [model.gWeight, model.gBias, model.gGamma, model.gBeta], \
										epochs = epochs, optimizer_config=model.optimizer_config)
	
	f5 = time.time()
	# print(f"Weight Update duration: {f5 - f4:.4f}")

	model.Weight, model.Bias, model.Gamma, model.Beta = new_weights
	model.optimizer_config = optims
	# f6 = time.time()
	model.get_weights()
	# f7 = time.time()
	# print(f"Get weight for the model duration: {f7 - f6:.4f}")
	
	model.get_grads()
	# f8 = time.time()
	# print(f"Get gradients for the model duration: {f8 - f7:.4f}")

f9 = time.time()
print(f"Total for 10 epochs: {f9 - f0:.4f}")
	# print(f"[{epoch}/{epochs}], Loss: {model.loss.item()}")
