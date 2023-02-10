import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def binary_tv_distance(p, q) -> float:
    return abs(p - q)


def calculate_tv_distance(answer_probs) -> float:
    tv_distances = []
    for row in answer_probs:
        tv_distances = [binary_tv_distance(p, q) for p, q in itertools.combinations(row, 2)]
        mean_tv_distance = np.array(tv_distances).mean()
        tv_distances.append(mean_tv_distance)
    return np.array(tv_distances).mean()


def load_results(path: Path) -> Dict[str, Any]:
    results = json.load(path.open('r'))
    results['metrics']['expected_tv_distance'] = calculate_tv_distance(np.array(results['metrics']['answer_probs']))
    return results['metrics']


def load_results_for(data_path: Path, test_name: str, model: str):
    return load_results(data_path / test_name / model / 'completions.json')


class MetricsPlotter:
    def __init__(self, test_names: List[str], model_names: List[str], data_path: Path = Path('persona_consistency'), save_dir_name: str = 'figures', ) -> None:
        self.data_path = data_path
        self.save_path = data_path / save_dir_name
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.test_names = test_names
        self.model_names = model_names
        for tn, mn in itertools.product(test_names, model_names):
            (self.save_path / tn / mn).mkdir(exist_ok=True, parents=True)
        self.metrics = {
            (test_name, model): load_results_for(self.data_path, test_name, model)
            for test_name, model in itertools.product(test_names, model_names)
            if (data_path / test_name / model).exists()
        }

    def get_metric_for_model(self, model_name: str, metric_name: str):
        return [self.metrics[(tn, model_name)][metric_name] for tn in self.test_names]

    def get_metric_for_task(self, task_name: str, metric_name: str):
        return [self.metrics[(task_name, model_name)][metric_name] for model_name in self.model_names]

    def get_default_scores(self, task_name: str):
        return [(np.array(self.metrics[(task_name, model_name)]['answer_probs'])[:, 0] > 0.5).mean() for model_name in self.model_names]

    def plot_expected_tv_distance_of_personas(self):
        plt.close('all')
        plt.clf()

        N = len(self.test_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = 0.6 / M
        if N > 1:
            offsets = np.linspace(-1, 1, M) * width * (M - 1) / 2
        else:
            offsets = [0]
        bars = []

        for i, model_name in enumerate(self.model_names):
            pos = ind + offsets[i]
            bars.append(plt.barh(pos, self.get_metric_for_model(model_name, 'expected_tv_distance'), width))

        plt.yticks(ind, self.test_names)
        plt.legend(bars, self.model_names)
        plt.title('Expected total variational distance of persona responses')
        plt.xlabel('Expected total variational distance')
        plt.tight_layout()
        plt.savefig(self.save_path / 'persona_tv_distances.png')

    def plot_mean_mae(self):
        plt.close('all')
        plt.clf()
        N = len(self.test_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = 0.6 / M
        if N > 1:
            offsets = np.linspace(-1, 1, M) * width * (M - 1) / 2
        else:
            offsets = [0]

        bars = []

        for i, model_name in enumerate(self.model_names):
            pos = ind + offsets[i]
            bars.append(plt.barh(pos, self.get_metric_for_model(model_name, 'mean_mae_on_questions'), width))

        plt.yticks(ind, self.test_names)
        plt.legend(bars, self.model_names)
        plt.title('MAE between default persona answers and induced answers')
        plt.xlabel('MAE')
        plt.tight_layout()
        plt.savefig(self.save_path / 'mae_of_default_persona.png')

    def overall_performance(self, task_name: str):
        plt.close('all')
        plt.clf()
        persona_scores = np.array(self.get_metric_for_task(task_name, 'persona_scores'))
        x = list(range(1, 1+len(self.model_names)))
        plt.violinplot(persona_scores.T)
        defaults = plt.plot(x, self.get_default_scores(task_name), 'o', markersize=12)
        plt.ylim([0,1])
        plt.xticks(x, labels=self.model_names, rotation=45)
        plt.ylabel('Fraction of answers matching behavior')
        plt.legend(defaults, ['Default Persona'])
        plt.title(f'Influence of persona on {task_name}')
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / f'overall_performance.png')

    def score_heat_map(self, model_name: str, task_name: str):
        plt.close('all')
        plt.clf()
        answer_probs = np.array(self.metrics[(task_name, model_name)]['answer_probs'])
        fig, ax = plt.subplots()
        im = ax.imshow(answer_probs, cmap='PiYG', interpolation='nearest')
        ax.set_xlabel('Personas')
        ax.set_ylabel('Questions')
        plt.title(f'{task_name}, {model_name}')
        cbar = ax.figure.colorbar(im, ax=ax)
        im.set_clim(0, 1)
        cbar.set_label('Behavior matching quality probability')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            top=False,
            labelleft=False,
            labelbottom=False
        )
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / model_name / 'scores.png')


    def diff_heat_map(self, model_name: str, task_name: str):
        plt.close('all')
        plt.clf()
        answer_probs = np.array(self.metrics[(task_name, model_name)]['answer_probs'])
        diff = answer_probs[:, 1:] - answer_probs[:, :1]
        fig, ax = plt.subplots()
        im = ax.imshow(diff, cmap='PiYG', interpolation='nearest')
        ax.set_xlabel('Personas')
        ax.set_ylabel('Questions')
        plt.title(f'{task_name}, {model_name}')
        cbar = ax.figure.colorbar(im, ax=ax)
        im.set_clim(-1, 1)
        cbar.set_label('Induced change in behavior')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            top=False,
            labelleft=False,
            labelbottom=False
        )
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / model_name / 'persona_shifts.png')

    def predictive_accuracy(self):
        plt.close('all')
        plt.clf()
        N = len(self.test_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = .75 / M
        if N > 1:
            offsets = np.linspace(-1, 1, M) * width * (M - 1) / 2
        else:
            offsets = [0]
        bars = []

        for i, model_name in enumerate(self.model_names):
            pos = ind + offsets[i]
            bars.append(plt.barh(pos, self.get_metric_for_model(model_name, 'overall_prediction_accuracy'), width))

        plt.yticks(ind, self.test_names)
        plt.legend(bars, self.model_names)
        plt.title('Predictive accuracy of answers to single questions')
        plt.xlabel('Accuracy')
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.savefig(self.save_path / 'prediction_accuracy.png')


    def plot_all(self):
        self.plot_expected_tv_distance_of_personas()
        self.plot_mean_mae()
        self.predictive_accuracy()
        for task_name in self.test_names:
            self.overall_performance(task_name)
        for task_name, model_name in itertools.product(self.test_names, self.model_names):
            self.score_heat_map(model_name, task_name)
            self.diff_heat_map(model_name, task_name)