import itertools
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap


def binary_tv_distance(p, q) -> float:
    return abs(p - q)


def calculate_tv_distance(answer_probs) -> float:
    mean_tv_distances = []
    for row in answer_probs:
        tv_distances = [binary_tv_distance(p, q) for p, q in itertools.combinations(row, 2)]
        mean_tv_distance = np.array(tv_distances).mean()
        mean_tv_distances.append(mean_tv_distance)
    return np.array(mean_tv_distances).mean()


def load_results(path: Path) -> Dict[str, Any]:
    results = json.load(path.open('r'))
    results['metrics']['expected_tv_distance'] = calculate_tv_distance(np.array(results['metrics']['answer_probs']))
    return results['metrics']


def load_results_for(data_path: Path, test_name: str, model: str):
    return load_results(data_path / test_name / model / 'completions.json')


def sample_question_for(data_path: Path, test_name: str):
    results = json.load((data_path / test_name / 'davinci' / 'completions.json').open('r'))
    return results['questions'][0]


class MetricsPlotter:
    def __init__(self, task_names: List[str], model_names: List[str], data_path: Path = Path('persona_consistency'), save_dir_name: str = 'figures', ) -> None:
        self.data_path = data_path
        self.save_path = data_path / save_dir_name
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.task_names = task_names
        self.model_names = model_names
        for tn, mn in itertools.product(task_names, model_names):
            (self.save_path / tn / mn).mkdir(exist_ok=True, parents=True)
        self.sample_questions = {task_name: sample_question_for(data_path, task_name) for task_name in task_names}
        self.metrics = {
            (test_name, model): load_results_for(self.data_path, test_name, model)
            for test_name, model in itertools.product(task_names, model_names)
            if (data_path / test_name / model).exists()
        }

    def get_metric_for_model(self, model_name: str, metric_name: str):
        return [self.metrics[(tn, model_name)][metric_name] for tn in self.task_names]

    def get_metric_for_task(self, task_name: str, metric_name: str):
        return [self.metrics[(task_name, model_name)][metric_name] for model_name in self.model_names]

    def get_default_scores(self, task_name: str):
        return [(np.array(self.metrics[(task_name, model_name)]['answer_probs'])[:, 0] > 0.5).mean() for model_name in self.model_names]

    def plot_expected_tv_distance_of_personas(self):
        plt.close('all')
        plt.clf()
        plt.figure(figsize=(7, 10))
        N = len(self.task_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = 0.75 / M
        if N > 1:
            offsets = np.linspace(-1, 1, M) * width * (M - 1) / 2
        else:
            offsets = [0]
        bars = []
        cmap = get_cmap('viridis_r')
        color_is = np.linspace(0, 1, M)
        for i, (model_name, color_i) in enumerate(zip(self.model_names, color_is)):
            pos = ind + offsets[i]
            color = cmap(color_i)
            bars.append(plt.barh(pos, list(reversed(self.get_metric_for_model(model_name, 'expected_tv_distance'))), width, color=color, edgecolor='black', alpha=0.75))
        plt.yticks(ind, list(reversed(self.task_names)))
        plt.legend(list(reversed(bars)), list(reversed(self.model_names)), bbox_to_anchor=(.75, 1.0), loc='upper left', facecolor='white', framealpha=1)
        ylim = (ind[0] - 0.5, ind[-1] + 0.5)
        plt.ylim(ylim)
        plt.title('Variation of persona responses')
        plt.xlabel('Expected Total Variational Distance')
        plt.tight_layout()
        plt.savefig(self.save_path / 'persona_tv_distances.png', bbox_inches='tight')

        data = {model: self.get_metric_for_model(model, 'expected_tv_distance') for model in self.model_names}
        df = pd.DataFrame.from_dict(data, columns=self.task_names, orient='index').T
        df.to_csv(self.save_path / 'persona_tv_distances.csv')


    def overall_performance(self, task_name: str):
        plt.close('all')
        plt.clf()
        persona_scores = np.array(self.get_metric_for_task(task_name, 'persona_scores'))
        x = list(range(1, 1+len(self.model_names)))
        violin = plt.violinplot(persona_scores.T)
        defaults = plt.plot(x, self.get_default_scores(task_name), 'o', markersize=12)
        plt.ylim([0,1])
        plt.xticks(x, labels=self.model_names, rotation=45, ha='right')
        plt.ylabel('Fraction of Positive Answers')
        plt.legend(defaults, ['Default Persona'])
        plt.title(f'Influence of Persona on {task_name}')
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
        cbar.set_label('Probability Assigned to Positive Answer')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,
            left=False,
            top=False,
            labelleft=False,
            labelbottom=True
        )
        example_text_lines = []
        for l in f'Example question from  {task_name}:\n{self.sample_questions[task_name]}'.split('\n'):
            example_text_lines.extend(textwrap.wrap(l, 27, break_long_words=True))
            example_text_lines.append('')
        text = '\n'.join(example_text_lines)
        plt.text(-13, 2, text, ha='left', va='top', fontsize=9)
        plt.xticks([0, 2.5, 6.5, 10.5, 14.5], labels=['default', 'poem', 'story', 'satire', 'opinion'], rotation=45, ha='right')
        ylim = (-0.5, 19.5)
        for x in [0.5, 4.5, 8.5, 12.5]:
            plt.vlines(x, *ylim, color='black')
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
        cbar.set_label('Change in Probability of Positive Answer')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,
            left=False,
            top=False,
            labelleft=False,
            labelbottom=True
        )
        plt.xticks([1.5, 5.5, 9.5, 13.5], labels=['poem', 'story', 'satire', 'opinion'], rotation=45, ha='right')
        ylim = (-0.5, 19.5)
        for x in [3.5, 7.5, 11.5]:
            plt.vlines(x, *ylim, color='black')
        example_text_lines = []
        for l in f'Example question from  {task_name}:\n{self.sample_questions[task_name]}'.split('\n'):
            example_text_lines.extend(textwrap.wrap(l, 27, break_long_words=True))
            example_text_lines.append('')
        text = '\n'.join(example_text_lines)
        plt.text(-13, 2, text, ha='left', va='top', fontsize=9)
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / model_name / 'persona_shifts.png')

    def predictive_accuracy_task(self, task_name):
        plt.close('all')
        plt.clf()
        accuracies = self.get_metric_for_task(task_name, 'overall_prediction_accuracy')
        x = list(range(1, 1+len(self.model_names)))
        xlim = (x[0] - 0.5, x[-1] + 0.5)
        plt.plot(x, accuracies, 'x-', label=task_name)
        plt.hlines([0.5], xlim[0], xlim[1], linestyles='dashed', colors=['gray'], label='Random Baseline')
        plt.xlim(xlim)
        plt.ylim([0,1])
        plt.xticks(x, labels=self.model_names, rotation=45, ha='right')
        plt.title(f'Prediction of induced persona behavior from default persona')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / 'prediction_accuracy.png')

    def expected_tv_distance_task(self, task_name):
        plt.close('all')
        plt.clf()
        accuracies = self.get_metric_for_task(task_name, 'expected_tv_distance')
        x = list(range(1, 1+len(self.model_names)))
        plt.plot(x, accuracies, 'x-', label=task_name)
        plt.ylim([-0.025,0.5])
        plt.xticks(x, labels=self.model_names, rotation=45, ha='right')
        plt.title(f'Variation of Personas Answers for {task_name}')
        plt.ylabel('Expected Total Variational Distance')
        xlim = (x[0] - 0.5, x[-1] + 0.5)
        plt.hlines([0.3333], xlim[0], xlim[1], linestyles='dashed', colors=['gray'], label='Random Behavior')
        plt.hlines([0], xlim[0], xlim[1], linestyles='dashed', colors=['blue'], label='Identical Behavior', alpha=0.5)
        plt.xlim(xlim)
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_path / task_name / 'expected_tv_distance.png')

    def predictive_accuracy(self):
        plt.close('all')
        plt.clf()
        plt.figure(figsize=(7, 10))
        N = len(self.task_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = .75 / M
        if N > 1:
            offsets = np.linspace(-1, 1, M) * width * (M - 1) / 2
        else:
            offsets = [0]
        bars = []

        cmap = get_cmap('viridis_r')
        color_is = np.linspace(0, 1, M)
        for i, (model_name, color_i) in enumerate(zip(self.model_names, color_is)):
            color = cmap(color_i)
            pos = ind + offsets[i]
            bars.append(plt.barh(pos, list(reversed(self.get_metric_for_model(model_name, 'overall_prediction_accuracy'))), width, color=color, edgecolor='black', alpha=0.75))

        ylim = (ind[0] - 0.5, ind[-1] + 0.5)

        plt.yticks(ind, list(reversed(self.task_names)))
        vline = plt.vlines(0.5, *ylim, linestyles='dashed', label='Random Baseline', color='black')
        plt.legend(list(reversed(bars)) + [vline], list(reversed(self.model_names)) + ['Random Baseline'], bbox_to_anchor=(0.8, 0.75), loc='upper left', facecolor='white', framealpha=1)
        plt.title('Prediction of induced persona behavior from default persona')
        plt.xlabel('Accuracy')
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(self.save_path / 'prediction_accuracy.png', bbox_inches='tight')

        data = {model: self.get_metric_for_model(model, 'overall_prediction_accuracy') for model in self.model_names}
        df = pd.DataFrame.from_dict(data, columns=self.task_names, orient='index').T
        df.to_csv(self.save_path / 'prediction_accuracy.csv')

    def plot_all(self):
        self.plot_expected_tv_distance_of_personas()
        self.predictive_accuracy()
        for task_name in self.task_names:
            self.overall_performance(task_name)
            self.predictive_accuracy_task(task_name)
            self.expected_tv_distance_task(task_name)
        for task_name, model_name in itertools.product(self.task_names, self.model_names):
            self.score_heat_map(model_name, task_name)
            self.diff_heat_map(model_name, task_name)