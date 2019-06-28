import torch
from torch import nn


class Trainer(object):

    def __init__(self, model, data_loader, device):
        self.data_loader = data_loader
        self.device = device
        self.model = model.to(self.device)

    def train(self, epochs, lr):

        print('Training on device: ', self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        calc_loss = nn.CrossEntropyLoss()
        steps = len(self.data_loader)
        for epoch in range(epochs):
            for step, (X_, Y_) in enumerate(self.data_loader):
                X, Y = X_.to(self.device), Y_.to(self.device)
                preds = self.model(X)
                loss = calc_loss(preds, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step+1)%100 == 0:
                    print("Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}".format(epoch+1, epochs,
                                                                           (step+1), steps,
                                                                           loss.item()))

        return self.model