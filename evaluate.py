import itertools
import json
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

import numpy as np

from persona_consistency.completion import (GPT3, AnswerSet, Completer,
                                            ThrottledCompleter,
                                            answer_multiple_choice)
from persona_consistency.datasets import load_evals_questions
from persona_consistency.plots import Metrics
from persona_consistency.prompting import SingleWordAnswerPrompter
from persona_consistency.questions import MultipleChoiceQuestion, YesNoQuestion


def add_response_to(question: MultipleChoiceQuestion, inducer: str) -> MultipleChoiceQuestion:
    """
    Add a response inducer to a given multiple-choice question.

    Args:
        question (MultipleChoiceQuestion): The original question.
        inducer (str): The inducer text to add.

    Returns:
        MultipleChoiceQuestion: The modified question with the inducer.
    """
    text = f"""{inducer}

Human:
{question.question_text}"""
    if isinstance(question, YesNoQuestion):
        return YesNoQuestion(text, question.label)
    return MultipleChoiceQuestion(
        text=text,
        options=question.options,
        label=question.label
    )


def evaluate_task(completer: Completer, simulation_inducers: List[str], questions: List[MultipleChoiceQuestion]) -> List[AnswerSet]:
    """
    Evaluate a task by answering multiple-choice questions.

    Args:
        completer (Completer): The completer object to use for answering questions.
        simulation_inducers (List[str]): List of simulation inducers.
        questions (List[MultipleChoiceQuestion]): List of questions to be answered.

    Returns:
        List[AnswerSet]: List of answer sets for the given questions.
    """
    prompter = SingleWordAnswerPrompter(lambda q: f'Human: {q.text}\n\nAssistant:', score_prefix='')
    answers = []
    for question in questions:
        question_variations = [question] + [add_response_to(question, inducer) for inducer in simulation_inducers]
        answers.append(answer_multiple_choice(question_variations, prompter, completer))
    return answers


def get_answer_probs(answers: AnswerSet) -> np.ndarray:
    """
    Get the probabilities of the answers.

    Args:
        answers (AnswerSet): The set of answers.

    Returns:
        np.ndarray: An array of answer probabilities.
    """
    return np.array([logits[q.label] for q, logits in zip(answers.questions, answers.mc_answer_probs)])  # type: ignore


def calculate_metrics(answers: List[AnswerSet]) -> Dict[str, Any]:
    """
    Calculate metrics for a list of answer sets.

    Args:
        answers (List[AnswerSet]): List of answer sets.

    Returns:
        Dict[str, Any]: Dictionary containing the calculated metrics.
    """
    answer_probs = np.stack([get_answer_probs(a) for a in answers])
    metrics = Metrics(answer_probs)
    return metrics.to_json()


def get_questions_path(task_name: str):
    """
    Get the path to the questions for a given task name.

    Args:
        task_name (str): The name of the task.

    Returns:
        Path: The path to the questions.

    Raises:
        NotImplementedError: If the task name is not found.
    """
    evals_path = Path(f'questions/Anthropic-evals')
    for category_path in evals_path.iterdir():
        for task_path in category_path.iterdir():
            if task_path.stem == task_name:
                return task_path
    raise NotImplementedError()


def evaluate(model_name: str, task_name: str, simulation_inducers: List[str], save_path: Path, n_questions: int):
    """
    Evaluate a model on a specific task and save the results.

    Args:
        model_name (str): The name of the model to evaluate.
        task_name (str): The name of the task to evaluate.
        simulation_inducers (List[str]): List of simulation inducers.
        save_path (Path): Path to save the results.
        n_questions (int): Number of questions to evaluate.

    Returns:
        None
    """
    save_dir = save_path / task_name / model_name
    save_dir.mkdir(exist_ok=True, parents=True)

    completer = ThrottledCompleter(GPT3(model_name), completion_batch_size=20, pause_seconds=1)

    eval_questions = load_evals_questions(get_questions_path(task_name))
    eval_questions = [q for q in eval_questions if len(q.options) == 2]
    if task_name in ['subscribes-to-Buddhism', 'subscribes-to-deontology']:
        eval_questions = [q for q in eval_questions if q.label == 'Yes']
    eval_questions = eval_questions[:n_questions]

    answers = evaluate_task(completer, simulation_inducers, eval_questions)
    
    metrics = calculate_metrics(answers)
    pprint(metrics)

    data = {}
    data['dataset'] = task_name
    data['metrics'] = metrics
    data['completions'] = {q.text: a.to_dict() for q, a in zip(eval_questions, answers)}
    data['personas'] = simulation_inducers
    data['questions'] = [q.text for q in eval_questions]

    with open(save_dir / 'completions.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    """
    Main function to run the evaluation on multiple models and tasks.

    Returns:
        None
    """
    save_path = Path('results') / '2_tasks'
    models = ['davinci', 'davinci-instruct-beta', 'text-davinci-001', 'text-davinci-002', 'text-davinci-003']
    tasks = ['conscientiousness', 'subscribes-to-deontology']
    max_personas = 19
    n_questions = 100

    with open('persona_inducers/text-davinci-003.json', 'r') as f:
        simulation_inducers = [f"{a['prompt']}\nAssistant: {a['completion'].strip()}" for a in json.load(f)['answers']][:max_personas]

    for model, task in itertools.product(models, tasks):
        evaluate(model_name=model, task_name=task, simulation_inducers=simulation_inducers, save_path=save_path, n_questions=n_questions)


if __name__ == "__main__":
    main()