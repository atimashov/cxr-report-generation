import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from argparse import  ArgumentParser

class chexpert_small(Dataset):
	"""
	TODO: add some description
	TODO: should we add gender, age, frontal / lateral, AP/PA to our NN or it should be different models?
	TODO: how to treat uncertain? 0.5?
	"""
	def __init__(self, root = '/Users/amelia/Downloads', input_size = 224, train = True):
		self.root = root
		self.input_size = input_size
		self.train = train
		self.df = pd.read_csv('{}/CheXpert-v1.0-small/{}'.format(self.root, 'train.csv' if self.train else 'valid.csv'))
		self.classes = [
			'No Finding', 'Enlarged Cardiomediastinum',	'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',	'Edema',
			'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
			'Support Devices'
		]

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		# load images and targets
		img_path = '{}/{}'.format(self.root, self.df['Path'][idx])
		img = Image.open(img_path).convert("RGB")
		# create targets: 1 - positive; everything else - 0;
		targets = torch.tensor([
			self.df[col][idx] if self.df[col][idx] in [0., 1.] else 0.5 if self.df[col][idx] == -1. else 0 for col in self.classes])

		if self.train:
			trans = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomCrop(self.input_size),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
			])
		else:
			trans = transforms.Compose([
				transforms.Resize((self.input_size, self.input_size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		return trans(img), targets

class iu_xray(Dataset):
	"""
	TODO: add some description
	TODO: will work only for inference / doesn't group by id (2 types of projections)
	"""
	def __init__(self, root = '/atlas/u/timashov/datasets/cxr/iu_cxr/', input_size = 224):
		self.root = root
		self.input_size = input_size
		self.df = pd.read_csv('{}/indiana_projections.csv'.format(self.root))

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		# load images and targets
		img_path = '{}images/images_normalized/{}'.format(self.root, self.df['filename'][idx])
		img = Image.open(img_path).convert("RGB")
		# TODO: projection is ignored at the moment;
		img_id = '{}images/images_normalized/{}'.format(self.root, self.df['uid'][idx])


		trans = transforms.Compose([
			transforms.Resize((self.input_size, self.input_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		return trans(img), img_id



def test():
	parser = ArgumentParser()
	parser.add_argument('--dataset', type=str, default='chexpert', help='which dataset to test')
	inputs = parser.parse_args()

	if inputs.dataset == 'chexpert':
		data = chexpert_small(root = '/atlas/u/timashov/datasets/cxr')
	elif inputs.dataset == 'iu_cxr':
		data = iu_xray()
	data_loader = DataLoader(
		data, batch_size = 4, shuffle = True, num_workers = 2, drop_last=True, pin_memory = True # TODO: increase number of workers
	)
	loop = tqdm(data_loader, leave = True)
	for batch_idx, (imgs, labels) in enumerate(loop):
		loop.set_postfix(imgs_shape=imgs.shape, lables_shape = labels.shape)

if __name__=='__main__':
	test()
