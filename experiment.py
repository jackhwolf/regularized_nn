import matplotlib.pyplot as plt
import numpy as np
from time import time
from data import Data
from model import Model

class Experiment:

    def __init__(self, **params):
        self.data = Data(**params)
        self.model = Model(**params)
        self.report = None

    def run(self):
        self.learn_training_sample()
        report = {}
        report.update(self.report_testing())
        report.update(self.report_sparsity())
        report.update(self.graph_interpolation())
        self.report = report
        return report        

    def learn_training_sample(self):
        for x, y in self.data.training_iterator():
            self.model.learn(x, y)

    def report_testing(self):
        true, pred = [], []
        for x, y in self.data.testing_iterator():
            true.append(y)
            pred.append(self.model.predict(x))
        true, pred = np.array(true), np.array(pred)
        mseloss = np.sum(np.power(true - pred, 2))
        acc = (true == np.sign(pred)).sum() / len(pred)
        return {'pred_loss': mseloss, 'pred_acc': acc}

    def report_sparsity(self):
        return {'sparsity': self.model.sparsity()}

    def graph_interpolation(self):
        fig, ax = plt.subplots()
        preds = []
        for i in range(self.data.N):
            preds.append(self.model.predict(self.data[i][0]))
        fig, ax = plt.subplots()
        cbar = ax.scatter(self.data.X[:,0], self.data.X[:,1], c=preds)
        fig.colorbar(cbar)
        path = f"Files/Results/{int(time()*1000)}.png"
        fig.savefig(path)
        return {'plot_path': path}


if __name__ == '__main__':
    exp = Experiment()
    exp.run()
    print(exp.report)