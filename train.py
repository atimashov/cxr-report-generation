from torch.nn.init import kaiming_normal_
import torch
from torch import optim
import os
from torch.utils.data import DataLoader
import numpy as np
from time import time, sleep
from datetime import datetime
from termcolor import colored
from argparse import  ArgumentParser
from utils import init_model, get_metrics, print_report
from datasets import chexpert_small
from loss import multi_label_loss
from tqdm import tqdm


def train_my(loader, model, epochs = 3, device = None, loss_func = multi_label_loss()):
	# init optimizer
	optimizer = optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 1e-5)

	for epoch in range(epochs):
		t = time()
		print_report(part = 'start', epoch = epoch)

		tacc, vacc = 0, 0
		tloss, vloss = 0, 0
		num_samples = 0

		val_acc_cl, val_acc, val_loss = get_metrics(loader = loader['val'], model = model, device = device, dtype = dtype)
		print('*****', val_acc_cl)


		loop = tqdm(loader['train'], leave = True)
		for idx, (imgs, labels) in enumerate(loop):
			model.train()  # put model to training mode
			imgs = imgs.to(device = device, dtype = dtype)
			labels = labels.to(device = device, dtype = dtype)

			scores = model(imgs)
			loss = loss_func(scores, labels)

			# Zero out all of the gradients for the variables which the optimizer will update.
			optimizer.zero_grad()

			# Backwards pass and computing gradients
			loss.backward()
			optimizer.step()

			# display
			loop.set_postfix(current_loss=loss.item())

		# print metrics
		train_acc_cl, train_acc, train_loss = get_metrics(loader = loader['train'], model = model, device = device, dtype = dtype)

		val_acc_cl, val_acc, val_loss = get_metrics(loader = loader['val'], model = model, device = device, dtype = dtype)
		print('*****', val_acc_cl)
		metrics = train_loss, val_loss, train_acc, val_acc
		print_report(part='accuracy', metrics=metrics)

		# TODO: create checkpoint
		torch.save(model.state_dict(), 'checkpoints/epoch{}_{}_{}.pt'.format(epoch, round(val_loss, 3), round(val_acc, 2)))

		# print time
		print_report(part = 'end', t = int(time() - t))
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