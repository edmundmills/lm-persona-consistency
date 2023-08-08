import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from persona_consistency.questions import (FreeResponseQuestion,
                                           MultipleChoiceQuestion,
                                           YesNoQuestion)


def make_evals_question(entry: Dict) -> MultipleChoiceQuestion:
    """
    Creates a MultipleChoiceQuestion object from a dictionary entry.

    Args:
        entry (Dict): A dictionary containing the question text, choices, and answer matching behavior.

    Returns:
        MultipleChoiceQuestion: A multiple-choice question object created from the entry.

    Raises:
        NotImplementedError: If the entry does not contain recognizable question format.
    """
    if 'Choices:' in entry['question']:
        question_text, choices = entry['question'].split('Choices:')
        question_text = question_text.strip()
        if 'Answer:' in choices:
            choices = choices.split('Answer:')[0].strip()
        options = {c.strip()[1]: c.strip()[4:] for c in choices.strip().split('\n') if c.strip()}
        label = entry['answer_matching_behavior'].strip()[1]
        return MultipleChoiceQuestion(question_text, options, label)
    elif 'Yes' in entry['answer_matching_behavior'] or 'No' in entry['answer_matching_behavior']:
        question_text = entry['question']
        label = 'Yes' if 'Yes' in entry['answer_matching_behavior'] else 'No'
        return YesNoQuestion(question_text, label)
    else:
        raise NotImplementedError()


def load_evals_questions(path: Path) -> List[MultipleChoiceQuestion]:
    """
    Loads multiple-choice questions from a file.

    Args:
        path (Path): The path to the file containing the questions in JSON format.

    Returns:
        List[MultipleChoiceQuestion]: A list of multiple-choice questions loaded from the file.
    """
    return [make_evals_question(json.loads(l)) for l in path.read_text().splitlines()]


def replace_brackets(template: str, replacer: str):
    """
    Replaces brackets in a template string with a specified replacer string.

    Args:
        template (str): The template string containing brackets to be replaced.
        replacer (str): The string to replace the brackets with.

    Returns:
        str: The modified string with brackets replaced.
    """
    return re.sub(r'\[[^)]*?\]', replacer, template)


def load_simulation_inducing_questions(path: Path, n_topics: Optional[int] = None) -> List[FreeResponseQuestion]:
    """
    Loads simulation-inducing questions from a file.

    Args:
        path (Path): The path to the file containing the questions and topics in JSON format.
        n_topics (Optional[int]): The number of topics to consider (if None, all topics are considered).

    Returns:
        List[FreeResponseQuestion]: A list of free-response questions loaded from the file.
    """
    task_data = json.loads((path).read_text())
    topics = task_data['topics']
    questions = task_data['questions']
    max_questions = min(len(questions), len(topics))
    return [
        FreeResponseQuestion(f'{replace_brackets(template, topic)}:')
        for template, topic
        in zip(questions[:max_questions], topics[:max_questions])
    ]
