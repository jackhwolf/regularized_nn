import numpy as np
from numpy.random import sample

class Data:

    def __init__(self, N=10, tr_sample=0.2, **kw):
        self.N = int(N)
        self.X = np.random.uniform(low=-1, high=1, size=(N, 2))
        self.Y = [self.fx(self.X[i,:]) for i in range(N)]
        self.tr_sample = float(tr_sample)
        self.tr_mask = np.array([False] * N)
        self.tr_mask[:int(self.N*self.tr_sample)] = True
        np.random.shuffle(self.tr_mask) 

    def training_iterator(self):
        for i in np.where(self.tr_mask)[0]:
            yield self[i]

    def testing_iterator(self):
        for i in np.where(~self.tr_mask)[0]:
            yield self[i]

    def fx(self, x):
        return np.sign(x[1] - (0.5*np.sin(np.pi*x[0])))

    def __getitem__(self, i):
        return (self.X[i,:], self.Y[i])

    def describe(self):
        return {'N': self.N, 'tr_sample': self.tr_sample}

def sample_graph():
    import matplotlib.pyplot as plt
    data = Data(5000)
    fig, ax = plt.subplots()
    cbar = ax.scatter(data.X[:,0], data.X[:,1], c=data.Y, cmap='bwr')
    ax.scatter(data.X[data.tr_mask,0], data.X[data.tr_mask,1], marker='x', c='k')
    fig.colorbar(cbar)
    ax.set_title("y = sign(x[1] - 0.5*sin(pi*x[0]))")
    fig.savefig("sample_data.png")

if __name__ == '__main__':
    sample_graph()