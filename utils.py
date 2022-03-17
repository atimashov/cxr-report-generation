import random
import numpy as np
from termcolor import colored
import torch
from torchvision.models import efficientnet_b5
import torch.nn as nn
from torch.nn import functional as F
from loss import multi_label_loss

def init_model(n_classes = 14, pretrained = True):
    model = efficientnet_b5(pretrained = pretrained)  # params.transfer
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
    num_correct_cl = np.zeros(14)
    num_samples = 0
    model.eval()  # set model to evaluation mode
    losses = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            # move to device, e.g. GPU
            imgs = imgs.to(device = device, dtype = dtype)
            labels = labels.to(device = device, dtype = dtype)

            preds = model(imgs)
            loss = loss_func(preds, labels)
            losses.append(loss.item())

            preds = torch.sigmoid(preds)
            preds[preds < 0.35] = 0.
            preds[preds.ge(0.35) & preds.le(0.65)] = 0.5
            preds[preds > 0.65] = 1.

            # TODO: calculate accuracy per class
            num_correct += (preds == labels).sum()
            num_correct_cl += (preds == labels).sum(dim = 0).cpu().detach().numpy()
            num_samples += preds.shape[0] * preds.shape[1]

            if i * imgs.shape[0] >= max_num:
                break
        acc = float(num_correct) / num_samples
        acc_cl = 14 * num_correct_cl / float(num_samples)
        loss = sum(losses) / len(losses)
    return acc_cl, acc, loss


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

        # Accuracy
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, y0, label, len_mask, steps, word_2_id):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    generated = []

    y1 = y0
    y2 = y0
    reps, _ = model.representation(x, label)
    # print(reps)

    for k in range(steps):
        logits, _, pred = model.decode(reps, y0, None, label)

        # logits = logits[:,0,:]
        # print("logits shape:", logits.shape)
        # print("input shape:", y0.shape)
        assert (logits.shape[1] == y0.shape[1])
        logits = logits[:, -1, :]

        # logits[:, y1] = float('-inf')
        # logits[:, y2] = float('-inf')

        y2 = y1
        y1 = torch.argmax(logits, dim=1).unsqueeze(0)
        generated.append(y1.item())

        # if y1.item() == word_2_id['<eos>']:
        #    break;

        # print("y0 shape:", y0.shape)
        # print("y1 shape:", y1.shape)
        y0 = torch.cat((y0, y1), dim=1)

    return generated
