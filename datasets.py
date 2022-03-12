import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import  ArgumentParser
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision
from torchvision import transforms


TRAIN_SIZE = 0.9
SEQUENCE_LEN = 1500

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
		img_id = self.df['uid'][idx]


		trans = transforms.Compose([
			transforms.Resize((self.input_size, self.input_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		return trans(img), img_id

def generate_vocabulary(df):
	df = df.fillna('')

	words = []
	for (report, finding) in zip(df.impression, df.findings):
		rp = re.findall(r"[\w']+|[.,!?;]", report)
		fg = re.findall(r"[\w']+|[.,!?;]", finding)
		words.append(rp)
		words.append(fg)
	# words.append(str.split(report))

	vocab = [x for sublist in words for x in sublist]

	vocab = sorted(np.unique(vocab))

	word_2_id = {}
	id_2_word = {}
	for ind, word in enumerate(vocab):
		word_2_id[str(word)] = ind
		id_2_word[ind] = word

	vocab_size = len(vocab)
	word_2_id['<eos>'] = vocab_size
	id_2_word[vocab_size] = '<eos>'
	word_2_id['<start>'] = vocab_size + 1
	id_2_word[vocab_size + 1] = '<start>'

	# print(word_2_id)
	# print(id_2_word)
	print("len vocab:", len(vocab))
	return word_2_id, id_2_word


def generate_vocabulary_mimic(df):
	df = df.fillna('')

	words = []
	for (report) in df.report:
		rp = re.findall(r"[\w']+|[.,!?;]", report)
		words.append(rp)

	vocab = [x for sublist in words for x in sublist]
	# vocab = [x[:7] for x in vocab]

	vocab = sorted(np.unique(vocab))

	word_2_id = {}
	id_2_word = {}
	for ind, word in enumerate(vocab):
		word_2_id[str(word)] = ind
		id_2_word[ind] = word

	vocab_size = len(vocab)
	word_2_id['<eos>'] = vocab_size
	id_2_word[vocab_size] = '<eos>'
	word_2_id['<start>'] = vocab_size + 1
	id_2_word[vocab_size + 1] = '<start>'

	# print(word_2_id)
	# print(id_2_word)
	return word_2_id, id_2_word


def get_train_val_df(df):
	"""
      Separates the dataframe into training and validation sets. Splits by subject id.
    """
	train_split = TRAIN_SIZE
	ids = df.id.unique()
	np.random.seed(1)
	train_uids = np.random.choice(ids, size=int(len(ids) * train_split), replace=False)

	df['in_train'] = None
	df['in_train'] = df["id"].apply(lambda x: x in train_uids)
	train_df = df[df['in_train'] == True]
	val_df = df[df['in_train'] == False]

	return train_df, val_df


class chestXRayDataset(Dataset):
	def __init__(self, df, img_dir, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word, tokenizer):

		self.block_size = block_size
		self.img_enc_width = img_enc_width
		self.img_enc_height = img_enc_height
		self.tokenizer = tokenizer

		df = df.reset_index()
		self.img_labels = df[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', \
							  'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', \
							  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']]
		# self.img_labels = None
		self.word_2_id = word_2_id
		self.id_2_word = id_2_word

		if len(self.img_labels) > 0:
			self.img_labels = self.img_labels.fillna(2)
			self.img_labels = self.img_labels + 1
			self.num_labels = len(self.img_labels.columns)
			self.img_labels = self.img_labels.to_numpy()

		df.report = df.report.fillna('')
		self.report = df.report.apply(lambda x: x.replace("\n", " "))
		self.report = self.report.apply(lambda x: re.sub(r'\W +', ' ', x))
		self.report = self.report.apply(lambda x: re.findall(r"[\w']+|[.,!?;]", x))
		# self.report = self.report.apply(lambda x: tokenizer.encode(str(x)).ids)
		self.report = self.report.apply(lambda x: [self.word_2_id[str(word)] for word in x])
		self.report = self.report.apply(
			lambda x: [word_2_id['<start>']] + x + [word_2_id['<eos>']] + [word_2_id['<eos>']])
		self.report = self.report.apply(lambda x: x if len(x) < block_size else x[:block_size])
		self.report_len = self.report.apply(lambda x: len(x))
		self.report = self.report.apply(
			lambda x: np.pad(x, (0, block_size - len(x)), constant_values=(word_2_id['<start>'], word_2_id['<eos>'])))
		# self.report = self.report.apply(lambda x: np.pad(x, (0,block_size-len(x)), constant_values=(0,3)))

		self.img_files = df['image_path']

	# for item in self.report:
	#    print(item)

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, idx):
		with Image.open(self.img_files[idx]) as image:
			image.load()
		width, height = image.size
		# image = image.resize((self.img_enc_height, self.img_enc_height))
		image = np.asarray(image)
		image2 = np.array(image, dtype=float)
		image = image2 / 255.0

		if len(image.shape) > 2:
			image = image[:, :, 0]

		m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
		preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.CenterCrop(min(width, height)),
			transforms.Resize((self.img_enc_height, self.img_enc_width)),
			transforms.Normalize(mean=m, std=s),
		])
		image = preprocess(image).squeeze(0).type(torch.FloatTensor)

		report = self.report[idx]
		len_mask = [False if i < self.report_len[idx] else True for i in range(self.block_size)]

		labels = self.img_labels[idx].tolist()
		# labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]

		return image, torch.LongTensor(report), torch.BoolTensor(len_mask), torch.IntTensor(labels)

	def collate_fn(self, samples):
		images, reports, len_masks, labels = [], [], [], []
		for image, report, len_mask, label in samples:
			images.append(image)
			reports.append(report)
			len_masks.append(len_mask)
			labels.append(label)

		# reports = pad_sequence(reports, batch_first=True, padding_value=self.word_2_id['<eos>'])
		reports = pad_sequence(reports, batch_first=True)
		images = pad_sequence(images, batch_first=True)
		len_masks = torch.vstack(len_masks)
		labels = torch.vstack(labels)
		return images, reports, len_masks, labels

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
