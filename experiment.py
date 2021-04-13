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
        self.report = None
        os.makedirs('Files/Plots', exist_ok=True)

    def run(self):
        self.learn_training_sample()
        report = {'timestamp': str(time())}
        report.update(self.report_testing())
        report.update(self.report_sparsity())
        report.update(self.graph_interpolation())
        report.update(self.data.describe())
        report.update(self.model.describe())
        self.report = report
        return report        

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
        return {'pred_sq_err': mseloss, 'pred_acc': acc}

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
        path = f"Files/Plots/{int(time()*1000)}.png"
        ax.set_title("Model interpolation")
        fig.savefig(path)
        return {'plot_path': path}


if __name__ == '__main__':
    import yaml
    import sys
    with open(sys.argv[1]) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    exp = Experiment(**content)
    exp.run()
    exp.save()
    print(exp.report)