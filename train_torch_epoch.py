import torch
import argparse
from models_torch.torch_original_compat import *
import time
from torch.utils.data import DataLoader
from dataset_config.factory import *
from dataset_config.roidb import *
from dataset_config.imdb import *
from dataset_config.pascal_voc import *
import warnings
import tqdm
from colorama import Fore, Back, Style
from util.data_util import check_dataset
from dataset_config.yolo_eval import *
from validate import test_for_train
warnings.filterwarnings(action="ignore")
import os
from loguru import logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logger with DEBUG level
logger.remove()  # Remove default handler
logger.add(
    sink=lambda msg: print(msg),
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)
# Add file handler for logging to file
logger.add(
    sink="logs/training_{time}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="1 week"
)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=500, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int,
                        help="specy the starting epoch")
    parser.add_argument('--batch_size', dest='batch_size',
                        default=8, type=int)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        default='custom', type=str,
                        help='choose dataset type pass custom if using other than voc dataset')
    parser.add_argument('--data', type=str, dest='data',
                        default="data.yaml", help='Give the path of custom data .yaml file' )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=False, type=bool)
    parser.add_argument('--display_interval', dest='display_interval',
                        default=20, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        default=False, type=bool)
    parser.add_argument('--save_interval', dest='save_interval',
                        default=10, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=True, action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--weights', default='', dest='weights',
                        help='provide the path of weight file (.pth) if resume')
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch',
                        default=100, type=int)
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)
    parser.add_argument('--device', default="0", dest='device', type=str,
                        help='Choose a gpu device 0, 1, 2 etc.')
    parser.add_argument('--savePath', default='results',
                        help='')
    parser.add_argument('--imgSize', default='1280,720',
                        help='image size w,h of image in your data') 
    parser.add_argument('--cleaning', dest='cleaning', 
                        default=False, type=bool,
                        help='Set true to remove small objects')
    parser.add_argument('--pix_th', dest='pix_th', 
                        default=12, type=int,
                        help='Pixel Threshold value') 
    parser.add_argument('--asp_th', dest='asp_th', 
                        default=1.5, type=float,
                        help='Aspect Ratio threshold')
    args, unknown = parser.parse_known_args()
    return args

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
		# gbias = torch.clamp(gbias, -1, 1)
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

	initial_lr = 0.001  # Initial learning rate
	warmup_epochs = 5   # Number of epochs for warmup
	plateau_epochs = 40 # Number of epochs for plateau phase
	decay_rate = 0.98    # Decay rate

	if epochs < warmup_epochs:
		learning_rate = initial_lr * (epochs) / warmup_epochs
		# logger.debug(f"Learning rate on epoch {epochs}: {learning_rate}")
	elif epochs < plateau_epochs:
		learning_rate = initial_lr
		# logger.debug(f"Learning rate on epoch {epochs}: {learning_rate}")
	else:
		learning_rate = initial_lr * (decay_rate ** (epochs - plateau_epochs))
		# logger.debug(f"Learning rate on epoch {epochs}: {learning_rate}")
	# i = 0

	for i in range(8):
		weight[i] = weight[i].cuda()
		gamma[i] = gamma[i].cuda()
		beta[i] = beta[i].cuda()
		gweight[i] = gweight[i].cuda()
		ggamma[i] = ggamma[i].cuda()
		gbeta[i] = gbeta[i].cuda()
	weight[8] = weight[8].cuda()
	gweight[8] = gweight[8].cuda()
	bias = bias.cuda()
	gbias = gbias.cuda()

	config = {'learning_rate': learning_rate, 'momentum': 0.9}
	with torch.no_grad():
		for i in range(8):
			k = i + 1
			config = optimizer_config['conv{}.weight'.format(k)]
			config['learning_rate'] = learning_rate
			weight[i], next_config = sgd_momentum(weight[i], gweight[i], config)
			optimizer_config['conv{}.weight'.format(k)] = next_config

		# for i in range(8):
			config = optimizer_config['bn{}.weight'.format(k)]
			config['learning_rate'] = learning_rate
			gamma[i], next_config = sgd_momentum(gamma[i], ggamma[i].reshape(-1), config)
			optimizer_config['bn{}.weight'.format(k)] = next_config
		
		# for i in range(8):
			config = optimizer_config['bn{}.bias'.format(k)]
			config['learning_rate'] = learning_rate
			beta[i], next_config = sgd_momentum(beta[i], gbeta[i].reshape(-1), config)
			optimizer_config['bn{}.bias'.format(k)] = next_config

		config = optimizer_config['conv9.0.weight']
		config['learning_rate'] = learning_rate
		weight[8], next_config = sgd_momentum(weight[8], gweight[8], config)
		optimizer_config['conv9.0.weight'] = next_config

		config = optimizer_config['conv9.0.bias']
		config['learning_rate'] = learning_rate
		bias, next_config = sgd_momentum(bias, gbias, config)
		optimizer_config['conv9.0.bias'] = next_config
	return (weight, bias, gamma, beta), optimizer_config

def drawBox(label:np.array, img:np.ndarray):
    # for i in range(label.shape[0]):
    h, w, _ = img.shape
    box = [label[0], label[1], label[2], label[3]]
    img = cv2.rectangle(img,(int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0,0,255), 1)
    return img

def showImg(img, labels, std=None, mean=None):
    # Convert the tensor to a numpy array
    _image = img
    image_np = _image.numpy().transpose((1, 2, 0))
    # image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)*255
    _img = Image.fromarray(image_np.astype('uint8'), 'RGB')
    _img = np.array(_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
    for i in range(labels.shape[0]):
        label = labels[i].numpy()
        _img = drawBox(label, _img)
    cv2.imshow('', _img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def train():
	logger.debug(f'Initializing training process')
	# Load Model
	args = parse_args()
	# logger.debug(f'Parsed arguments: {args}')
	os.environ["CUDA_VISIBLE_DEVICES"] = args.device
	logger.debug(f'Using GPU device: {args.device}')

	# Load BDD Dataset
	args.scaleCrop = False 
	logger.debug(f'Loading dataset from {args.data}')
	data_dict = check_dataset(args.data)
	train_path, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
	nc = int(data_dict['nc'])  # number of classes
	names = data_dict['names']  # class names
	logger.debug(f'Dataset contains {nc} classes: {names}')
	assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {args.data}'  # check
	logger.info(f'Loading training data from {train_path}')
	train_dataset = Custom_yolo_dataset(train_path,
										cleaning=args.cleaning,
										pix_th=args.pix_th,
										asp_th=args.asp_th,
										scale_Crop=args.scaleCrop)
	args.val_dir = val_dir
	if not args.data_limit==0:
		logger.debug(f'Limiting dataset to {args.data_limit} samples')
		train_dataset = torch.utils.data.Subset(train_dataset, range(0, args.data_limit))
	logger.info(f'Dataset loaded successfully')
	logger.info(f'Training Dataset size: {len(train_dataset)}')
	
	logger.debug(f'Creating DataLoader with batch_size={args.batch_size}, num_workers={args.num_workers}')
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,    #args.batch_size
									shuffle=True, num_workers=args.num_workers,      # args.num_workers
									collate_fn=detection_collate, drop_last=True, pin_memory=True)

	# Load Model
	model = Pytorch_bn(num_classes=nc)
	logger.debug('Model initialized')
	
	if args.resume:
		checkpoint = torch.load(args.weights, weights_only=False)
		logger.debug('Loaded pretrained checkpoint')
		pytorch_model = checkpoint['model']

		try:
			for param, val in model.modtorch_model.params.items():
				for param1, val1 in pytorch_model.items():
					if (param == param1):
						model.modtorch_model.params[param] = val1.cuda()
			logger.debug('Transferred pretrained weights to model')
		except Exception as e:
			logger.debug(f'Error transferring pretrained weights to model: {e}')
			raise e
	
	# sample code
	epochs = args.max_epochs
	f0 = time.time()
	iters_per_epoch = int(len(train_dataset) / args.batch_size)
	logger.info(f"Training for {epochs} epochs with {iters_per_epoch} iterations per epoch")

	_output_dir = os.path.join(os.getcwd(), args.output_dir)
	if not os.path.exists(_output_dir):
		logger.debug(f"Creating output directory: {_output_dir}")
		os.makedirs(_output_dir, exist_ok=True)
	logger.debug(f"Output directory: {_output_dir}")

	# performance tracking parameters
	max_map = 0
	# logger.debug("Initialized performance tracking parameters")
	
	for epoch in range(args.start_epoch, epochs):
		logger.info(f"Starting epoch {epoch+1}/{epochs}")
		epoch_start_time = time.time()
		loss_temp = 0
		train_data_iter = iter(train_dataloader)
		
		for step in tqdm.tqdm(range(iters_per_epoch), desc=f'Epoch {epoch}', total=iters_per_epoch):
			if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
				scale_index = np.random.randint(*cfg.scale_range)
				cfg.input_size = cfg.input_sizes[scale_index]
				# logger.debug(f"Multi-scale training: Changed input size to {cfg.input_size}")
				
			im_data, gt_boxes, gt_classes, num_obj, im_info = next(train_data_iter)
			# logger.debug(f"Batch {step+1}/{iters_per_epoch}: Image shape: {im_data.shape}, Objects: {num_obj.sum().item()}")

			# for i in range(im_data.shape[0]):
			# 	showImg(im_data[i], gt_boxes[i])

			im_data = im_data.cuda()
			gt_boxes = gt_boxes.cuda()
			gt_classes = gt_classes.cuda()
			num_obj = num_obj.cuda()

			# Forward
			model.out, model.cache, model.FOut = model.modtorch_model.Forward(im_data)
			# logger.debug(f"Forward pass completed for batch {step+1}")

			# Loss Calculation
			model.loss, model.loss_grad = model.modtorch_model.loss(model.out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)
			# logger.debug(f"Loss calculated for batch {step+1}: {model.loss.item():.4f}")
			loss_temp += model.loss.item()
			# Backpropagation
			model.dout, model.grads = model.modtorch_model.backward(model.loss_grad, model.cache)
			model.get_weights()
			model.get_grads()
			# logger.debug(f"Backpropagation completed for batch {step+1}")

			# Weight Update
			new_weights, optims = sgd_momentum_update(Inputs = [model.Weight,  model.Bias,  model.Gamma,  model.Beta],
											gInputs = [model.gWeight, model.gBias, model.gGamma, model.gBeta], \
												epochs = epoch, optimizer_config=model.optimizer_config)
			
			model.Weight, model.Bias, model.Gamma, model.Beta = new_weights
			model.optimizer_config = optims
			model.load_weights(new_weights)
			# logger.debug(f"Weights updated for batch {step+1}")

		epoch_time = time.time() - epoch_start_time
		logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
		
		# Validation
		logger.info("Starting validation")
		mAP, _ = test_for_train(_output_dir, model, args, val_path, names)
		logger.info(f"Validation mAP: {mAP*100:.4f}")
		
		# Save mAP to file
		mAP_path = os.path.join(_output_dir, "mAP.txt")
		with open(mAP_path, mode="a+") as map_file: 
			map_file.write(f"mAP: {round((mAP * 100), 2)}%, loss: {round(loss_temp/iters_per_epoch, 4)}%\n")
		# logger.debug(f"mAP saved to {mAP_path}")
		
		# Save model checkpoint
		if mAP > max_map:
			max_map = mAP
			save_name_best = os.path.join(_output_dir, f'yolov2_best_map@{epoch}.pth')
			# save_name = os.path.join(_output_dir, f"yolov2tiny_epoch{epoch+1}.pth")
			logger.info(f'\n\t--------------------->>Saving best weights at Epoch {epoch}, with mAP={round((mAP*100),2)}% and loss={round(model.loss.item(),2)}')
			torch.save({
				'model': model.modtorch_model.params,
				'epoch': epoch + 1,
				'mAP': mAP
			}, save_name_best)
			logger.info(f"Model checkpoint saved to {save_name_best}")

		logger.info(f"Epoch {epoch}/{epochs} completed, Loss: {loss_temp/iters_per_epoch:.4f}")

		model.get_weights()
		
	logger.info(f"Training completed in {time.time() - f0:.2f} seconds")

if __name__ == "__main__":
	train()