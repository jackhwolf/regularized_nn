import torch
import torch

class Model(torch.nn.Module):

    def __init__(self, r1d=10, l2d=5, r2d=10, epochs=5, lr=1e-3, scale=0.1, regularization=2, **kw):
        super().__init__()
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.scale = float(scale)
        self.regularization = regularization
        self.regularization_method = self.regularize_1 if regularization == 1 else self.regularize_2
        self.r1d = int(r1d)
        self.l2d = int(l2d)
        self.r2d = int(r2d)
        self.relu_1 = torch.nn.Linear(2, self.r1d)
        self.lin_1 = torch.nn.Linear(self.r1d + 2, self.l2d)
        self.relu_2 = torch.nn.Linear(self.l2d, self.r2d)
        self.lin_2 = torch.nn.Linear(self.r2d + self.l2d, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)

    def learn(self, x, y):
        x, y = self._tensor(x), self._tensor([y])
        for e in range(self.epochs):
            pred = self.forward(x)
            loss = self.criterion(pred, y)
            loss += self.scale * self.regularization_method()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.detach().numpy().item()

    def predict(self, x):
        x = self._tensor(x)
        with torch.no_grad():
            return self.forward(x).detach().numpy().item()

    def forward(self, x):
        x = self._tensor(x)
        r1_out = self.relu_1(x).clamp(min=0)
        l1_out = self.lin_1(torch.cat([r1_out, x], 0))
        r2_out = self.relu_2(l1_out).clamp(min=0)
        l2_out = self.lin_2(torch.cat([r2_out, l1_out], 0))
        return l2_out

    def regularize_1(self):
        r = 0
        sumpow = lambda x: x.detach().pow(2).sum().numpy().item()
        r += sumpow(self.relu_1.weight)
        r += sumpow(self.lin_1.weight[:,:self.r1d])
        r += sumpow(self.relu_2.weight)
        r += sumpow(self.lin_2.weight[:,:self.r2d])
        return r

    def regularize_2(self):
        r = 0
        sumpow = lambda x: x.detach().pow(2).sum().numpy().item()
        powsumabs = lambda x: x.detach().abs().sum().pow(2).numpy().item()
        r += powsumabs(self.relu_1.weight)
        r += sumpow(self.lin_1.weight[:,:self.r1d])
        r += powsumabs(self.relu_2.weight)
        r += sumpow(self.lin_2.weight[:,:self.r2d])
        return r

    def sparsity(self):
        threshold = 1e-3
        sperc = lambda x: ((x < threshold).sum() / x.numel()).detach().numpy().item()
        sparsities = []
        sparsities.append(sperc(self.relu_1.weight))
        sparsities.append(sperc(self.lin_1.weight[:,:self.r1d]))
        sparsities.append(sperc(self.relu_2.weight))
        sparsities.append(sperc(self.lin_2.weight[:,:self.r2d]))
        return sparsities

    def _tensor(self, a):
        if not isinstance(a, torch.FloatTensor):
            a = torch.FloatTensor(a)
        return a

    def describe(self):
        out = {
            'epochs': self.epochs,
            'lr': self.lr,
            'scale': self.scale,
            'regularization': self.regularization
        }
        return out

if __name__ == '__main__':
    from data import Data
    d = Data(50)
    m = Model(4, 2, 4)
    x, y = d[0]
    print(m.learn(x, y))