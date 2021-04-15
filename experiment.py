import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
import json
from data import Data
from model import Model

class Experiment:

    def __init__(self, **params):
        self.data = Data(**params)
        self.model = Model(**params)
        self.report = {}
        os.makedirs('Files/Plots', exist_ok=True)

    def run(self):
        self.learn_training_sample()
        self.report['timestamp'] = str(time())
        self.report.update(self.report_testing())
        self.report.update(self.report_sparsity())
        self.report.update(self.graph_interpolation())
        self.report.update(self.data.describe())
        self.report.update(self.model.describe())
        self.save()
        return self.report.copy()        

    def save(self):
        path = 'Files/results.json'
        with open(path, 'a+') as fp:
            curr = fp.read()
            if curr == '':
                curr = []
            else:
                curr = json.loads(curr)
            curr.append(self.report)
            fp.write(json.dumps(curr, indent=4))

    def learn_training_sample(self):
        k = int(self.data.N * self.data.tr_sample)
        for i, (x, y) in enumerate(self.data.training_iterator()):
            self.model.learn(x, y)
            print(f"{i}/{k}")

    def report_testing(self):
        true, pred = [], []
        for x, y in self.data.testing_iterator():
            true.append(y)
            pred.append(self.model.predict(x))
        true, pred = np.array(true), np.array(pred)
        mseloss = np.sum(np.power(true - pred, 2))
        acc = (true == np.sign(pred)).sum() / len(pred)
        return {'pred_sq_err': mseloss, 'pred_acc': acc}

    def report_sparsity(self):
        return {'sparsity': self.model.sparsity()}

    def graph_interpolation(self):

        def make_param_text():
            text =  "Parameters:"
            text += f"\n-N: {self.data.N}"
            text += f"\n-Sample %: {self.data.tr_sample}"
            text += f"\n-Architecture: {self.model.r1d}, {self.model.l2d}, {self.model.r2d}"
            text += f"\n-Epochs: {self.model.epochs}"
            text += f"\n-LR: {self.model.lr}"
            text += f"\n-Reg. Scalar: {self.model.scale}"
            text += f"\n-Regularization: {self.model.regularization}"
            text += f"\n\nResult Stats:"
            text += f"\n-Prediction err.: {np.round(self.report['pred_sq_err'], 2)}"
            text += f"\n-Prediction acc.: {np.round(self.report['pred_acc'], 2)}"
            text += f"\n-Sparsity:"
            for i, s in enumerate(self.report['sparsity']):
                text += f"\n     -L{i}: {s}"
            return text

        fig, ax = plt.subplots()
        preds = []
        for i in range(self.data.N):
            preds.append(self.model.predict(self.data[i][0]))
        fig, ax = plt.subplots()
        cbar = ax.scatter(self.data.X[:,0], self.data.X[:,1], c=preds, cmap='bwr')
        ax.scatter(self.data.X[self.data.tr_mask,0], self.data.X[self.data.tr_mask,1], marker='x', c='k')
        fig.colorbar(cbar)
        path = f"Files/Plots/{int(time()*1000)}.png"
        ax.set_title("Model interpolation")
        ax.text(1.3, .5, make_param_text(),
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes
        )
        fig.savefig(path, bbox_inches='tight')
        return {'plot_path': path}


if __name__ == '__main__':
    import yaml
    import sys
    with open(sys.argv[1]) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    exp = Experiment(**content)
    exp.run()
    print(exp.report)