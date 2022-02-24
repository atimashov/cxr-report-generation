from torchvision.models import  efficientnet_b5
import torch
from loss import multi_label_loss
from termcolor import colored


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

def get_metrics(loader, model, device, dtype, loss_func = multi_label_loss(), max_num = 10000):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    losses = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            # move to device, e.g. GPU
            imgs = imgs.to(device = device, dtype = dtype)
            labels = labels.to(device = device, dtype = dtype)

            scores = model(imgs)
            loss = loss_func(scores, labels)
            losses.append(loss.item())

            # TODO: consider "not certain" later
            preds = scores.copy()
            preds[preds < 0.5] = 0.
            preds[preds >= 0.5] = 1.
            # TODO: calculate accuracy per class
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)

            if i * imgs.shape[0] >= max_num:
                break
        acc = 100 * float(num_correct) / num_samples
        loss = sum(losses) / len(losses)
    return acc, loss


def print_report(part, epoch = None, t = None, metrics = None):
    if part == 'start':
        print('{} Epoch {}{} {}'.format(' ' * 60, ' ' * (3 - len(str(epoch))), epoch, ' ' * 61))
        print(' ' * 132)
    elif part == 'end':
        t_min, t_sec = str(t // 60), str(t % 60)
        txt = 'It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec)
        print(txt)
        print()
        print(colored('-' * 132, 'cyan'))
        print()
    else: # report statistics
        # loss
        train_loss, val_loss, train_acc, val_acc = metrics
        t_loss, v_loss = round(train_loss, 3), round(val_loss, 3)
        prefix = 6 - str(t_loss).find('.')
        postfix = 4 - (len(str(t_loss)) - str(t_loss).find('.'))
        t_loss = '{}{}{}'.format(' ' * prefix, t_loss, '0' * postfix)
        prefix = 6 - str(v_loss).find('.')
        postfix = 4 - (len(str(v_loss)) - str(v_loss).find('.'))
        v_loss = '{}{}{}'.format(' ' * prefix, v_loss, '0' * postfix)
        t_print = 'TRAIN   : loss = {}'.format(t_loss)
        v_print = 'VALIDATE: loss = {}'.format(v_loss)

        # mAP@0.5
        t_acc, v_acc = round(100 * train_acc, 2), round(100 * val_acc, 2)
        if '.' not in str(t_acc):
            t_acc = '{}{}.00'.format((3 - len(str(t_acc))) * ' ', t_acc)
        else:
            prefix = 3 - str(t_acc).find('.')
            postfix = 2 - (len(str(t_acc)) - str(t_acc).find('.'))
            t_acc = '{}{}{}'.format(' ' * prefix, t_acc, '0' * postfix)

        if '.' not in str(v_acc):
            v_acc = '{}{}.00'.format((3 - len(str(v_acc))) * ' ', v_acc)
        else:
            prefix = 3 - str(v_acc).find('.')
            postfix = 2 - (len(str(v_acc)) - str(v_acc).find('.'))
            v_acc = '{}{}{}'.format(' ' * prefix, v_acc, '0' * postfix)

        t_print += ' | Accuracy  = {}%'.format(t_acc)
        v_print += ' | Accuracy  = {}%'.format(v_acc)

        print(t_print)
        print(v_print)