import torch
import torch.nn as nn

class multi_label_loss(nn.Module):
    def __init__(self):
        super(multi_label_loss, self).__init__()

    def forward(self, predictions, target):
        predictions = torch.sigmoid(predictions)
        print('***', predictions.dtype, target.dtype)
        loss_fn = nn.BCELoss()

        loss = loss_fn(predictions, target)

        return torch.mean(loss)

def test():
    target = torch.ones((13, 14))
    predictions = torch.randn((13, 14))
    print(target.shape, predictions.shape)
    loss_func = multi_label_loss()
    loss = loss_func(predictions, target)
    print(loss)

if __name__=='__main__':
	test()