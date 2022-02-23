from torch.nn.init import kaiming_normal_
import torch
from torch import optim
import os
from torch.utils.data import DataLoader
import numpy as np
from time import time, sleep
from datetime import datetime
# from termcolor import colored
from argparse import  ArgumentParser
from utils import init_model
from datasets import chexpert_small
from loss import multi_label_loss
from tqdm import tqdm


def train_my(loader, model, epochs = 3, device = None, loss_func = multi_label_loss()):
	# init optimizer
	optimizer = optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 1e-5)

	for epoch in range(epochs):
		# t = time()
		# print(colored('-' * 50, 'cyan'))
		# print(colored('{} Epoch {}{} {}'.format('-' * 20, ' ' * (2 - len(str(epoch))), epoch, '-' * 20), 'cyan'))
		# print(colored('-' * 50, 'cyan'))
		#
		# tacc, vacc = 0, 0
		# tloss, vloss = 0, 0
		# num_samples = 0

		steps = 0
		loop = tqdm(loader['train'], leave = True)
		for idx, (imgs, labels) in enumerate(loop):
			model.train()  # put model to training mode
			imgs = imgs.to(device = device, dtype = dtype)
			labels = labels.to(device = device, dtype = torch.long)

			scores = model(imgs)
			loss = loss_func(scores, labels)

			# Zero out all of the gradients for the variables which the optimizer will update.
			optimizer.zero_grad()

			# Backwards pass and computing gradients
			loss.backward()
			optimizer.step()

			# display
			loop.set_postfix(imgs_shape=loss.item())


		# create checkpoint


		# t = int(time() - t)
		# t_min, t_sec = str(t // 60), str(t % 60)
		# print(colored('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec), 'cyan'))
		# print(colored('-' * 50, 'cyan'))
		# print()

if __name__ == '__main__':
	parser = ArgumentParser()
	# parser.add_argument('--model', type=str, default='my_alexnet', help='model name: my_alexnet, alexnet, my_vgg16, vgg16')
	parser.add_argument('--dataset-path', type=str, default='/atlas/u/timashov/datasets/cxr', help='path to dataset: image_net_10, corrosion_dataset')
	parser.add_argument('--n-print', type=int, default=50, help='how often to print')
	parser.add_argument('--n-epochs', type=int, default=1000, help='number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='batch size')
	# parser.add_argument('--transfer', type=str, default='False', help='transfer/full learning')
	parser.add_argument('--use-gpu', type=str, default='True', help='gpu/cpu')
	inputs = parser.parse_args()
	print(inputs)

	# inputs.transfer = True if inputs.transfer == 'True' else False
	USE_GPU = True if inputs.use_gpu == 'True' else False
	device = torch.device('cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu')
	dtype = torch.float32  # TODO: find out how it affects speed and accuracy

	# run model
	n_classes = 14
	model = init_model(n_classes).to(device)

	# create data loaders
	dataset_path = inputs.dataset_path
	data_train = chexpert_small(root = dataset_path, train = True)
	data_val = chexpert_small(root = dataset_path, train = False)

	data_loader = {
		'train': DataLoader(
			data_train, batch_size = inputs.batch_size, shuffle = True, num_workers = 6, drop_last=True, pin_memory=True
		),
		'val': DataLoader(data_val, batch_size = inputs.batch_size, shuffle = False, num_workers = 6)
	}

	# run training
	train_my(data_loader, model, epochs = inputs.n_epochs, device = device)