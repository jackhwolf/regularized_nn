import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
import json
from sklearn.model_selection import ParameterGrid
from data import Data
from model import Model

class ExperimentBase:

    def __init__(self, rp, data, model):
        self.data = data
        self.model = model
        self.report = {}
        self.results_path = rp
        os.makedirs('Files/Plots', exist_ok=True)

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
        s = self.model.sparsity()
        avg, sd = np.mean(s), np.std(s)
        out = {
            'sparsity': s,
            'sparsity_avg': avg,
            'sparsity_sd': sd
        }
        return out

    def run(self):
        pass


    def save(self):
        path = self.results_path
        with open(path, 'a+') as fp:
            curr = fp.read()
            if curr == '':
                curr = []
            else:
                curr = json.loads(curr)
            curr.append(self.report)
            fp.write(json.dumps(curr, indent=4))

class InterpolationExperiment(ExperimentBase):

    def __init__(self, **params):
        super().__init__('Files/interpolation_results.json', Data(**params), Model(**params))

    def run(self, graph=True):
        self.learn_training_sample()
        self.report['timestamp'] = str(time())
        self.report.update(self.report_testing())
        self.report.update(self.report_sparsity())
        if graph:
            self.report.update(self.graph_interpolation())
        self.report.update(self.data.describe())
        self.report.update(self.model.describe())
        self.save()
        return self.report.copy()        

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
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        path = f"Files/Plots/{int(time()*1000)}.png"
        x = np.linspace(-1, 1, 100)
        x, y = np.meshgrid(x, x)
        x, y = x.flatten(), y.flatten()
        z = []
        for i in range(len(x)):
            z.append(self.model.predict(np.array([x[i], y[i]])))
        cbar = ax.plot_surface(x.reshape(100,100), y.reshape(100,100), np.array(z).reshape(100,100), cmap='bwr', linewidth=0, antialiased=False)
        fig.colorbar(cbar, shrink=0.5, aspect=5)
        ax.set_title("Model interpolation")
        ax.text2D(1.3, .5, make_param_text(),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes
        )
        fig.savefig(path, bbox_inches='tight')
        return {'plot_path': path}

class SparsityExperiment(ExperimentBase):

    def __init__(self, **params):
        params = {k: [v] if not isinstance(v, list) else v for k, v in params.items()}
        self.params = list(ParameterGrid(params))
        super().__init__('Files/sparsity_results.json', Data(**self.params[0]), Model(**self.params[0]))
        self.report = {'params': [], 'reports': []}

    def run(self):
        for i in range(len(self.params)):
            self.model = Model(**self.params[i])
            self.learn_training_sample()
            rep = {}
            rep.update(self.report_testing())
            rep.update(self.report_sparsity())
            self.report['params'].append(self.params[i])
            self.report['reports'].append(rep)
        self.report['timestamp'] = str(time())
        self.report.update(self.graph_sparsity())
        self.save()

    def graph_sparsity(self):
        def make_param_text():
                text =  "Parameters:"
                text += f"\n-N: {self.data.N}"
                text += f"\n-Sample %: {self.data.tr_sample}"
                text += f"\n-Architecture: {self.model.r1d}, {self.model.l2d}, {self.model.r2d}"
                text += f"\n-LR: {self.model.lr}"
                text += f"\n-Reg. Scalar: {self.model.scale}"
                text += f"\n-Regularization: {self.model.regularization}"
                return text
        fig, ax = plt.subplots()
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Average layer sparsity")
        path = f"Files/Plots/{int(time()*1000)}.png"
        x, y, s = [], [], []
        for i in range(len(self.params)):
            param, rep = self.report['params'][i], self.report['reports'][i]
            x.append(param['epochs'])
            y.append(rep['sparsity_avg'])
            s.append(rep['sparsity_sd'])
        print(x)
        print(y)
        ax.errorbar(x, y, s)
        ax.set_title("Avg. Sparsity vs. Training Epochs")
        ax.text(1.1, .5, make_param_text(),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes
        )
        fig.savefig(path, bbox_inches='tight')
        return {'plot_path': path}

if __name__ == '__main__':
    import yaml
    import sys
    with open(sys.argv[1]) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    exp = {
        'interpolation': InterpolationExperiment,
        'sparsity': SparsityExperiment,
    }
    exp = exp[content['experiment']](**content)
    exp.run()
    print(exp.report)