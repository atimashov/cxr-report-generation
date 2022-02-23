from torchvision.models import  efficientnet_b5
import torch


def init_model(n_classes = 14):
    model = efficientnet_b5(pretrained = True)  # params.transfer
    # requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, n_classes)
    # model = model.to(device=device)
    return model

# def init_weights(model):
# 	for layer in model.features:
# 		if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
# 			kaiming_normal_(layer.weight)
# 	for layer in model.classifier:
# 		if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
# 			kaiming_normal_(layer.weight)
# 	return model
#
# def get_accuracy(loader, model, device, loss_func = torch.nn.CrossEntropyLoss()):
# 	num_correct = 0
# 	num_samples = 0
# 	model.eval()  # set model to evaluation mode
# 	losses = []
# 	with torch.no_grad():
# 		for (imgs, labels) in loader:
# 			imgs = imgs.to(device = device, dtype = dtype)  # move to device, e.g. GPU
# 			labels = labels.to(device = device, dtype = torch.long)
#
# 			scores = model(imgs)
# 			loss = loss_func(scores, labels)
# 			losses.append(float(loss))
#
# 			_, preds = scores.max(1)
# 			num_correct += (preds == labels).sum()
# 			num_samples += preds.size(0)
#
# 		acc = 100 * float(num_correct) / num_samples
# 		loss = sum(losses) / len(losses)
# 	return acc, loss
#
# def get_optimizer(old_params, model, optimizer = None):
# 	prev_lr, prev_wd, prev_do, prev_opt, flag = old_params
# 	if optimizer is None:
# 		if prev_do != 50:
# 			for i, layer in enumerate(model.features):
# 				if type(layer) == torch.nn.Dropout:
# 					model.features[i] = torch.nn.Dropout(prev_do / 100)
# 			for i, layer in enumerate(model.classifier):
# 				if type(layer) == torch.nn.Dropout:
# 					model.classifier[i] = torch.nn.Dropout(prev_do / 100)
# 		if prev_opt == 'Adam':
# 			optimizer = optim.Adam(model.parameters(), lr = prev_lr, weight_decay = prev_wd)
# 		else:
# 			optimizer = optim.SGD(model.parameters(), lr = prev_lr, weight_decay = prev_wd, momentum = 0.9, nesterov = True)
# 	else:
# 		lr, wd, do, opt, flag = get_params()
# 		if flag:
# 			return (lr, wd, do, opt, flag), model, optimizer
#
# 		if (lr != prev_lr) or (wd != prev_wd) or (do != prev_do) or (opt != prev_opt):
# 			if prev_do != do:
# 				for i, layer in enumerate(model.features):
# 					if type(layer) == torch.nn.Dropout:
# 						model.features[i] = torch.nn.Dropout(prev_do / 100)
# 				for i, layer in enumerate(model.classifier):
# 					if type(layer) == torch.nn.Dropout:
# 						model.classifier[i] = torch.nn.Dropout(prev_do / 100)
# 			prev_lr, prev_wd, prev_do, prev_opt = lr, wd, do, opt
#
# 			if prev_opt == 'Adam':
# 				optimizer = optim.Adam(model.parameters(), lr=prev_lr, weight_decay=prev_wd)
# 			else:
# 				optimizer = optim.SGD(model.parameters(), lr=prev_lr, weight_decay=prev_wd, momentum=0.9, nesterov=True)
# 	return (prev_lr, prev_wd, prev_do, prev_opt, flag), model, optimizer


