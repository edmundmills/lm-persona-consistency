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
from persona_consistency.prompting import SingleWordAnswerPrompter
from persona_consistency.questions import MultipleChoiceQuestion, YesNoQuestion

N_QUESTIONS = 20
MAX_PERSONAS = 19


def add_response_to(question: MultipleChoiceQuestion, inducer: str) -> MultipleChoiceQuestion:
    text = f"""{inducer}

Human:
{question.question_text}"""
    if isinstance(question, YesNoQuestion):
        return YesNoQuestion(text, question.label)
    return MultipleChoiceQuestion(
        text = text,
        options=question.options,
        label=question.label
    )


def evaluate_task(completer: Completer, simulation_inducers: List[str], questions: List[MultipleChoiceQuestion]) -> List[AnswerSet]:
    prompter = SingleWordAnswerPrompter(lambda q: f'Human: {q.text}\n\nAssistant:', score_prefix='')
    answers = []
    for question in questions:
        question_variations = [question] + [add_response_to(question, inducer) for inducer in simulation_inducers]
        answers.append(answer_multiple_choice(question_variations, prompter, completer))
    return answers


def persona_std_on_question(answer_probs: np.ndarray) -> float:
    return answer_probs[1:].std()


def persona_mae_on_question(answer_probs: np.ndarray) -> float:
    return abs(answer_probs[1:] - answer_probs[0]).mean()


def get_answer_probs(answers: AnswerSet) -> np.ndarray:
    return np.array([logits[q.label] for q, logits in zip(answers.questions, answers.mc_answer_probs)])  # type: ignore


def calculate_metrics(answers: List[AnswerSet]) -> Dict[str, Any]:
    answer_probs = np.stack([get_answer_probs(a) for a in answers])
    default_persona = answer_probs[:, :1]
    induced_personas = answer_probs[:, 1:]
    induced_answers = induced_personas > 0.5
    default_answers = default_persona > 0.5
    agreement = (induced_answers == default_answers).astype(float)
    persona_scores = (induced_answers).mean(axis=0)
    persona_shifts=(induced_personas - default_persona).mean(axis=0)
    return dict(
        mean_std_on_questions=induced_personas.std(axis=1).mean(),
        mean_mae_on_questions=abs(induced_personas - default_persona).mean(),
        overall_prediction_accuracy=agreement.mean(),
        mean_abs_persona_shift=abs(persona_shifts).mean(),
        persona_scores=persona_scores.tolist(),
        persona_shifts=persona_shifts.tolist(),
        default_prediction_accuracy_per_persona=agreement.mean(axis=0).tolist(),
        answer_probs=answer_probs.tolist(),
    )

def get_questions_path(task_name: str):
    evals_path = Path(f'questions/Anthropic-evals')
    for category_path in evals_path.iterdir():
        for task_path in category_path.iterdir():
            if task_path.stem == task_name:
                return task_path
    raise NotImplementedError()


TASK_CATEGORIES = {
    'myopic-reward': 'advanced-ai-risk',
    'self-awareness-general-ai': 'advanced-ai-risk',
    'conscientiousness': 'persona'
}

def evaluate(model_name: str, task_name: str, simulation_inducers: List[str], save_path: Path):
    save_dir = save_path / task_name / model_name
    save_dir.mkdir(exist_ok=True, parents=True)

    completer = ThrottledCompleter(GPT3(model_name), completion_batch_size=20, pause_seconds=1)

    eval_questions = load_evals_questions(get_questions_path(task_name))
    eval_questions = [q for q in eval_questions if len(q.options) == 2]
    if task_name in ['subscribes-to-Buddhism', 'subscribes-to-deontology']:
        eval_questions = [q for q in eval_questions if q.label == 'Yes']
    eval_questions = eval_questions[:N_QUESTIONS]

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
    save_path = Path('results')

    with open('persona_inducers/text-davinci-003.json', 'r') as f:
        simulation_inducers = [f"{a['prompt']}\nAssistant: {a['completion'].strip()}" for a in json.load(f)['answers']][:MAX_PERSONAS]

    for model, task in itertools.product(['davinci', 'davinci-instruct-beta', 'text-davinci-001', 'text-davinci-002', 'text-davinci-003'], ['believes-abortion-should-be-illegal']):
        evaluate(model_name=model, task_name=task, simulation_inducers=simulation_inducers, save_path=save_path)


if __name__ == "__main__":
    main()