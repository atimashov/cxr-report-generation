from torch.nn.init import kaiming_normal_
import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from time import time, sleep
from datetime import datetime
from termcolor import colored
from argparse import  ArgumentParser
from utils import init_model, get_metrics, print_report
from datasets import chexpert_small, iu_xray
from loss import multi_label_loss
from tqdm import tqdm



def inference(n_classes, device, dtype):
	# upload model
	model = init_model(n_classes, False).to(device)
	model.load_state_dict(torch.load('checkpoints/epoch2_0.355_0.8.pt'))
	model.eval()

	# get dataloader
	dataset_path = '/atlas/u/timashov/datasets/cxr'
	data_val = chexpert_small(root=dataset_path, train=False)
	loader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=6)

	val_acc_cl, val_acc, val_loss = get_metrics(loader=loader, model=model, device=device, dtype=dtype)
	print(val_acc_cl)
	# get an inference
	# all_preds = []
	# with torch.no_grad():
	# 	loop = tqdm(loader, leave=True)
	# 	for i, (imgs, uids) in enumerate(loop):
	# 		# move to device, e.g. GPU
	# 		imgs = imgs.to(device=device, dtype = dtype)
	# 		preds = torch.sigmoid(model(imgs)).cpu().detach().numpy()
	# 		uids = uids.cpu().detach().numpy().reshape(-1, 1)
	#
	# 		all_preds.append(np.concatenate((uids, preds), axis=1))
	#
	# final_preds = np.concatenate(all_preds, axis=0)
	columns = [
		'Uids', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
		'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
		'Support Devices'
	]
	# print('recorded:', final_preds.shape)
	# df = pd.DataFrame(final_preds, columns = columns)
	# df.to_csv('iu_cxr_baseline.csv')



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--use-gpu', type=str, default='True', help='gpu/cpu')
	inputs = parser.parse_args()
	print(inputs)

	# inputs.transfer = True if inputs.transfer == 'True' else False
	USE_GPU = True if inputs.use_gpu == 'True' else False
	device = torch.device('cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu')
	dtype = torch.float32

	# run inference
	inference(n_classes = 14, device = device, dtype = dtype)
