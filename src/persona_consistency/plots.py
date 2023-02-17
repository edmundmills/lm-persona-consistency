import itertools
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from tqdm import tqdm


def binary_tv_distance(p, q) -> float:
    return abs(p - q)


def calculate_tv_distance(answer_probs) -> float:
    mean_tv_distances = []
    for row in answer_probs:
        tv_distances = [binary_tv_distance(p, q) for p, q in itertools.combinations_with_replacement(row, 2)]
        mean_tv_distance = np.array(tv_distances).mean()
        mean_tv_distances.append(mean_tv_distance)
    return np.array(mean_tv_distances).mean()


def load_results(path: Path) -> Dict[str, Any]:
    results = json.load(path.open('r'))
    return results['metrics']


def load_results_for(data_path: Path, test_name: str, model: str) -> "Metrics":
    metric_dict = load_results(data_path / test_name / model / 'completions.json')
    return Metrics(np.array(metric_dict['answer_probs']))


def sample_question_for(data_path: Path, test_name: str):
    results = json.load((data_path / test_name / 'davinci' / 'completions.json').open('r'))
    return results['questions'][0]


class Metrics:
    def __init__(self, answer_probs: np.ndarray) -> None:
        self._answer_probs = answer_probs
        self.metric_fns = dict(
            default_behavior=self.__class__.default_behavior.fget,
            persona_behavior=self.__class__.persona_behavior.fget,
            default_answers=self.__class__.default_answers.fget,
            persona_answers=self.__class__.persona_answers.fget,
            default_aggregate_score=self.__class__.default_aggregate_score.fget,
            persona_aggregate_scores=self.__class__.persona_aggregate_scores.fget,
            persona_shifts=self.__class__.persona_shifts.fget,
            overall_prediction_accuracy=self.__class__.overall_prediction_accuracy.fget,
            expected_tv_distance_personas=self.__class__.expected_tv_distance_personas.fget,
            expected_tv_distance_questions=self.__class__.expected_tv_distance_questions.fget,
            answer_probs=self.__class__.answer_probs.fget
        )

    def __getitem__(self, metric_name: str):
        return self.metric_fns[metric_name](self)

    def to_dict(self):
        return {k: v(self) for k, v in self.metric_fns.items()}

    def to_json(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.to_dict().items()}

    @property
    def answer_probs(self):
        return self._answer_probs

    @property
    def default_behavior(self):
        return self.answer_probs[:, :1]

    @property
    def persona_behavior(self):
        return self.answer_probs[:, 1:]

    @property
    def default_answers(self):
        return (self.default_behavior > 0.5)

    @property
    def persona_answers(self):
        return (self.persona_behavior > 0.5)

    @property
    def default_aggregate_score(self):
       return self.default_answers.mean()

    @property
    def persona_aggregate_scores(self):
        return self.persona_answers.mean(axis=0)

    @property
    def persona_shifts(self):
        return (self.persona_behavior - self.default_behavior).mean(axis=0)

    @property
    def overall_prediction_accuracy(self):
        return (self.persona_answers == self.default_answers).astype(float).mean()

    @property
    def expected_tv_distance_personas(self):
        return calculate_tv_distance(self.answer_probs)

    @property
    def expected_tv_distance_questions(self):
        return calculate_tv_distance(self.default_behavior.T)


class MetricsCollection:
    def __init__(self, metrics_dict: Dict[Tuple[str, str], Metrics], task_names: List[str], model_names: List[str]) -> None:
        self.metrics_dict = metrics_dict
        self.task_names = task_names
        self.model_names = model_names

    @classmethod
    def load(cls, task_names: List[str], model_names: List[str], data_path: Path = Path('persona_consistency')):
        metrics_collection_dict = {
            (test_name, model): load_results_for(data_path, test_name, model)
            for test_name, model in itertools.product(task_names, model_names)
            if (data_path / test_name / model).exists()
        }
        return cls(metrics_collection_dict, task_names, model_names)

    def __getitem__(self, task_and_model_name: Tuple[str, str]):
        return self.metrics_dict[task_and_model_name]

    def get_metric_for_model(self, model_name: str, metric_name: str) -> List[Any]:
        return [self[(tn, model_name)][metric_name] for tn in self.task_names]

    def get_metric_for_task(self, task_name: str, metric_name: str) -> List[Any]:
        return [self[(task_name, model_name)][metric_name] for model_name in self.model_names]


class MetricsPlotter:
    def __init__(self, task_names: List[str], model_names: List[str], data_path: Path = Path('persona_consistency'), save_dir_name: str = 'figures', ) -> None:
        self.data_path = data_path
        self.save_path = data_path / save_dir_name
        self.metrics_collection = MetricsCollection.load(task_names, model_names, data_path)
        self.sample_questions = {task_name: sample_question_for(data_path, task_name) for task_name in task_names}

    @property
    def task_names(self):
        return self.metrics_collection.task_names

    @property
    def model_names(self):
        return self.metrics_collection.model_names

    def get_metric_for_model(self, model_name: str, metric_name: str) -> List[Any]:
        return [self.metrics_collection[(tn, model_name)][metric_name] for tn in self.task_names]

    def get_metric_for_task(self, task_name: str, metric_name: str) -> List[Any]:
        return [self.metrics_collection[(task_name, model_name)][metric_name] for model_name in self.model_names]

    def plot_expected_tv_distance_of_personas(self):
        plt.close('all')
        plt.clf()
        plt.figure(figsize=(7, 10))
        N = len(self.task_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = 0.75 / M
        if M > 1:
            offsets = np.linspace(-1, 1, M) * width * (M - 1) / 2
        else:
            offsets = [0]
        bars = []
        cmap = get_cmap('viridis_r')
        color_is = np.linspace(0, 1, M)
        for i, (model_name, color_i) in enumerate(zip(self.model_names, color_is)):
            tv_distances = self.get_metric_for_model(model_name, 'expected_tv_distance_personas')
            pos = ind + offsets[i]
            color = cmap(color_i)
            bars.append(plt.barh(pos, list(reversed(tv_distances)), width, color=color, edgecolor='black', alpha=0.75))
        plt.yticks(ind, list(reversed(self.task_names)))
        plt.legend(list(reversed(bars)), list(reversed(self.model_names)), bbox_to_anchor=(.75, 1.0), loc='upper left', facecolor='white', framealpha=1)
        ylim = (ind[0] - 0.5, ind[-1] + 0.5)
        plt.ylim(ylim)
        plt.title('Behavior Variation Across Contexts')
        plt.xlabel('Expected Total Variation Distance')
        plt.tight_layout()
        plt.savefig(self.save_path / 'persona_tv_distances.png', bbox_inches='tight')

        data = {model: self.get_metric_for_model(model, 'expected_tv_distance_personas') for model in self.model_names}
        df = pd.DataFrame.from_dict(data, columns=self.task_names, orient='index').T
        df.to_csv(self.save_path / 'persona_tv_distances.csv')

    def overall_performance(self, task_name: str):
        plt.close('all')
        plt.clf()
        persona_scores = np.array(self.get_metric_for_task(task_name, 'persona_aggregate_scores'))
        x = list(range(1, 1+len(self.model_names)))
        violin = plt.violinplot(persona_scores.T)
        defaults = plt.plot(x, self.get_metric_for_task(task_name, 'default_aggregate_score'), 'o', markersize=12)
        plt.ylim([0,1])
        plt.xticks(x, labels=self.model_names, rotation=45, ha='right')
        plt.ylabel('Fraction of Positive Answers')
        plt.legend(defaults, ['Default Behavior'])
        plt.title(f'Influence of Context: {task_name}')
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / f'overall_performance.png')

    def score_heat_map(self, model_name: str, task_name: str):
        plt.close('all')
        plt.clf()
        answer_probs = np.array(self.metrics_collection[(task_name, model_name)]['answer_probs'])
        fig, ax = plt.subplots()
        im = ax.imshow(answer_probs, cmap='PiYG', interpolation='nearest')
        ax.set_xlabel('Contexts')
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
        for l in f'Example question from {task_name}:\n{self.sample_questions[task_name]}'.split('\n'):
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
        answer_probs = np.array(self.metrics_collection[(task_name, model_name)]['answer_probs'])
        diff = answer_probs[:, 1:] - answer_probs[:, :1]
        fig, ax = plt.subplots()
        im = ax.imshow(diff, cmap='PiYG', interpolation='nearest')
        ax.set_xlabel('Contexts')
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
        for l in f'Example question from {task_name}:\n{self.sample_questions[task_name]}'.split('\n'):
            example_text_lines.extend(textwrap.wrap(l, 27, break_long_words=True))
            example_text_lines.append('')
        text = '\n'.join(example_text_lines)
        plt.text(-13, 2, text, ha='left', va='top', fontsize=9)
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / model_name / 'persona_shifts.png')

    def predictive_accuracy_task(self, task_name):
        plt.close('all')
        plt.clf()
        default_scores = np.array(self.get_metric_for_task(task_name, 'default_aggregate_score'))
        predict_majority = np.maximum(default_scores, 1 - default_scores)
        accuracies = self.get_metric_for_task(task_name, 'overall_prediction_accuracy')
        x = list(range(1, 1+len(self.model_names)))
        xlim = (x[0] - 0.5, x[-1] + 0.5)
        plt.plot(x, accuracies, 'x-', label='Predict same answer as default answer')
        plt.plot(x, predict_majority, 'x-', label='Always predict overall tendency')
        plt.hlines([0.5], xlim[0], xlim[1], linestyles='dashed', colors=['gray'], label='Random Baseline')
        plt.xlim(xlim)
        plt.ylim([0,1])
        plt.xticks(x, labels=self.model_names, rotation=45, ha='right')
        plt.title(f'Predictive Power of Default Behavior: {task_name}')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / 'prediction_accuracy.png')

    def expected_tv_distance_task(self, task_name):
        plt.close('all')
        plt.clf()
        persona_tv_difference = self.get_metric_for_task(task_name, 'expected_tv_distance_personas')
        question_tv_difference = self.get_metric_for_task(task_name, 'expected_tv_distance_questions')
        x = list(range(1, 1+len(self.model_names)))
        plt.plot(x, persona_tv_difference, 'x-', label='Across all contexts')
        # plt.plot(x, persona_type_tv_difference, 'x-', label='Across context types')
        plt.plot(x, question_tv_difference, 'x-', label='Across questions')
        plt.ylim([-0.025,0.525])
        plt.xticks(x, labels=self.model_names, rotation=45, ha='right')
        plt.title(f'Variation of Behavior: {task_name}')
        plt.ylabel('Expected Total Variation Distance')
        xlim = (x[0] - 0.5, x[-1] + 0.5)
        plt.hlines([0.5], xlim[0], xlim[1], linestyles='dashed', colors=['green'], label='Behavior is 50% 0.0, 50% 1.0')
        plt.hlines([0.3333], xlim[0], xlim[1], linestyles='dashed', colors=['gray'], label='Behavior is uniform between [0, 1]')
        plt.hlines([0], xlim[0], xlim[1], linestyles='dashed', colors=['blue'], label='Behavior is always the same', alpha=0.5)
        plt.xlim(xlim)
        plt.legend(facecolor='white', framealpha=1)

        plt.tight_layout()
        plt.savefig(self.save_path / task_name / 'expected_tv_distance.png')

    def mean_behavior_shift_model(self, model_name: str):
        plt.close('all')
        plt.clf()
        data = np.array(self.get_metric_for_model(model_name, 'persona_shifts'))
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='PiYG', interpolation='nearest')
        ax.set_xlabel('Contexts')
        ax.set_ylabel('Tasks')
        plt.title(f'Context Induced Shift in Behavior: {model_name}')
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.33, aspect=30*0.33)
        im.set_clim(-1, 1)
        cbar.set_label('Mean Probability Shift')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,
            left=True,
            top=False,
            labelleft=True,
            labelbottom=True
        )
        n_rows = len(self.task_names)
        plt.xticks([1.5, 5.5, 9.5, 13.5], labels=['poem', 'story', 'satire', 'opinion'], rotation=45, ha='right')
        ylim = (-0.5, n_rows - 0.5)
        for x in [3.5, 7.5, 11.5]:
            plt.vlines(x, *ylim, color='black')
        plt.yticks(np.arange(n_rows), self.task_names)
        plt.tight_layout()
        (self.save_path / 'models' / model_name).mkdir(exist_ok=True, parents=True)
        plt.savefig(self.save_path / 'models' / model_name / 'persona_shifts.png', bbox_inches="tight")

    def mean_behavior_shift_task(self, task_name: str):
        plt.close('all')
        plt.clf()
        data = np.array(self.get_metric_for_task(task_name, 'persona_shifts'))
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='PiYG', interpolation='nearest')
        ax.set_xlabel('Models')
        ax.set_ylabel('Tasks')
        plt.title(f'Context Induced Shift in Behavior: {task_name}')
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.22, aspect=30*0.22)
        im.set_clim(-1, 1)
        cbar.set_label('Mean Probability Shift')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,
            left=True,
            top=False,
            labelleft=True,
            labelbottom=True
        )
        n_rows = len(self.model_names)
        plt.xticks([1.5, 5.5, 9.5, 13.5], labels=['poem', 'story', 'satire', 'opinion'], rotation=45, ha='right')
        ylim = (-0.5, n_rows - 0.5)
        for x in [3.5, 7.5, 11.5]:
            plt.vlines(x, *ylim, color='black')
        plt.yticks(np.arange(n_rows), self.model_names)
        plt.tight_layout()
        plt.savefig(self.save_path / task_name / 'persona_shifts.png', bbox_inches="tight")

    def predictive_accuracy(self):
        plt.close('all')
        plt.clf()
        plt.figure(figsize=(7, 10))
        N = len(self.task_names)
        M = len(self.model_names)
        ind = np.arange(N) 
        width = .75 / M
        if M > 1:
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
        plt.title('Prediction of Behavior from Default Behavior')
        plt.xlabel('Accuracy')
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(self.save_path / 'prediction_accuracy.png', bbox_inches='tight')

        data = {model: self.get_metric_for_model(model, 'overall_prediction_accuracy') for model in self.model_names}
        df = pd.DataFrame.from_dict(data, columns=self.task_names, orient='index').T
        df.to_csv(self.save_path / 'prediction_accuracy.csv')

    def plot_all(self):
        self.save_path.mkdir(exist_ok=True, parents=True)
        for tn, mn in itertools.product(self.task_names, self.model_names):
            (self.save_path / tn / mn).mkdir(exist_ok=True, parents=True)

        self.plot_expected_tv_distance_of_personas()
        self.predictive_accuracy()
        for model_name in tqdm(self.model_names, desc='Making Model Plots'):
            self.mean_behavior_shift_model(model_name)
        for task_name in tqdm(self.task_names, desc='Making Task Plots'):
            self.mean_behavior_shift_task(task_name)
            self.overall_performance(task_name)
            self.predictive_accuracy_task(task_name)
            self.expected_tv_distance_task(task_name)
        for task_name, model_name in tqdm(list(itertools.product(self.task_names, self.model_names)), desc='Making Model Plots'):
            self.score_heat_map(model_name, task_name)
            self.diff_heat_map(model_name, task_name)