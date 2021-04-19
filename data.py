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
    n = 500
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    path = "sample_data_3d.png"
    x = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, x)
    x, y = x.flatten(), y.flatten()
    z = []
    for i in range(len(x)):
        z.append(np.sign(y[i] - (0.5*np.sin(np.pi*x[i]))))
    cbar = ax.plot_surface(x.reshape(n,n), y.reshape(n,n), np.array(z).reshape(n,n), cmap='bwr', linewidth=0, antialiased=False)
    fig.colorbar(cbar, shrink=0.5, aspect=5)
    ax.set_title("y = sign(x[1] - 0.5*sin(pi*x))")
    fig.savefig(path, bbox_inches='tight')
    return {'plot_path': path}

if __name__ == '__main__':
    sample_graph()