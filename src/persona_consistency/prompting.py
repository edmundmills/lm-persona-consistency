import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

from persona_consistency.questions import (FreeResponseQuestion, Question,
                                           ScalarQuestion)

logger = logging.getLogger(__name__)


class Prompter(ABC):
    """
    Abstract base class for creating different types of prompts.

    Methods:
        make_prompt: Abstract method to create a prompt from a given question.
        get_answer: Abstract method to extract an answer from a given completion.
    """

    @abstractmethod
    def make_prompt(self, question: Question) -> str:
        """Create a prompt from the given question.
        
        Args:
            question (Question): The question object to create a prompt from.

        Returns:
            str: The created prompt.
        """

    @abstractmethod
    def get_answer(self, completion: str) -> Any:
        """Extract an answer from the given completion.
        
        Args:
            completion (str): The completion text to extract an answer from.

        Returns:
            Any: The extracted answer.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


def clean(score_text: str) -> str:
    """Clean the score text by removing unwanted characters.
    
    Args:
        score_text (str): The score text to clean.

    Returns:
        str: The cleaned score text.
    """
    score_text = score_text.strip().split()[0]
    if score_text[0] == '[':
        score_text = score_text[0:]
    if score_text[-1] in ':],':
        score_text = score_text[:-1]
    return score_text


def get_score_text(completion: str, score_prefix: str) -> str:
    """Extract the score text from the completion using the given prefix.
    
    Args:
        completion (str): The completion text to extract the score from.
        score_prefix (str): The prefix used to identify the score text.

    Returns:
        str: The extracted score text.
    """
    completion = completion.lower()
    score_prefix = score_prefix.lower()
    if score_prefix not in completion:
        logging.warning(f'Score prefix ({score_prefix}) not in completion: {completion}')
        return clean(completion)
    split_text = completion.split(score_prefix)
    if len(split_text) <= 1:
        logging.warning(f'Nothing after score prefix in completion: {completion}')
        return ''
    score_text = split_text[-1]
    return clean(score_text)
    

def scalar_score_from(score_text: str) -> Union[float, None]:
    """Convert the score text into a scalar score.
    
    Args:
        score_text (str): The score text to convert.

    Returns:
        Union[float, None]: The converted scalar score, or None if conversion fails.
    """
    if score_text[-1] in ',.;:':
        score_text = score_text[:-1]
    if score_text[-1] == '%':
        percent = True
        score_text = score_text[:-1]
    else:
        percent = False
    try:
        score = float(score_text)
    except:
        return None
    if percent:
        score /= 100
    return score


def normalize(score: float, range: Tuple[float, float]) -> float:
    """Normalize the given score within the specified range.
    
    Args:
        score (float): The score to normalize.
        range (Tuple[float, float]): The range within which to normalize the score.

    Returns:
        float: The normalized score.
    """
    score = min(score, range[1])
    score = max(score, range[0])
    return (score - range[0]) / (range[1] - range[0])


class SingleWordAnswerPrompter(Prompter):
    """A prompter that creates prompts for single-word answers.

    Attributes:
        prompt_fn (Callable): A function to create prompts from questions.
        score_prefix (str): The prefix used to identify the score text.
    """
    def __init__(
            self,
            prompt_fn: Callable[[Question], str],
            score_prefix: str):
        self.prompt_fn = prompt_fn
        self.score_prefix = score_prefix

    def make_prompt(self, question: Question) -> str:
        return self.prompt_fn(question)

    def get_answer(self, completion: str) -> Any:
        return get_score_text(completion, self.score_prefix)


class ReasoningWithScorePrompter(Prompter):
    """A prompter that creates prompts for reasoning with scores.

    Attributes:
        prompt_fn (Callable): A function to create prompts from scalar questions.
        score_prefix (str): The prefix used to identify the score text.
        normalize_to_range (Optional[Tuple[float, float]]): The range to normalize the score to.
    """
    def __init__(
            self,
            prompt_fn: Callable[[ScalarQuestion], str],
            score_prefix: str,
            range_to_normalize_from: Optional[Tuple[float, float]] = None):
        self.prompt_fn = prompt_fn
        self.score_prefix = score_prefix
        if range_to_normalize_from and not (range_to_normalize_from[0] < range_to_normalize_from[1]):
            raise ValueError(f'Normalization range must be from low to high, got {range_to_normalize_from}')
        self.normalize_to_range = range_to_normalize_from

    def make_prompt(self, question: Question) -> str:
        assert isinstance(question, ScalarQuestion)
        return self.prompt_fn(question)

    def get_answer(self, completion: str) -> Any:
        score_text = get_score_text(completion, self.score_prefix)
        score = scalar_score_from(score_text)
        if score is not None and self.normalize_to_range:
            score = normalize(score, self.normalize_to_range)
        return score


class FreeResponsePrompter(Prompter):
    """A prompter that creates prompts for free-response questions.

    Attributes:
        prompt_fn (Optional[Callable]): A function to create prompts from free-response questions.
    """
    def __init__(self, prompt_fn: Optional[Callable[[FreeResponseQuestion], str]] = None):
        self.prompt_fn = prompt_fn if prompt_fn is not None else lambda q: q.text
 
    def make_prompt(self, question: Question):
        assert isinstance(question, FreeResponseQuestion)
        return self.prompt_fn(question)
    
    def get_answer(self, completion: str):
        return completion


class ListPrompter(Prompter):
    """A prompter that creates prompts for list-based questions.

    Attributes:
        bullet (str): The bullet character used to separate list items.
    """
    def __init__(self, bullet: str = '-') -> None:
        self.bullet = f'\n{bullet}'
    
    def make_prompt(self, question: Question) -> str:
        return f'{question.text.strip()}{self.bullet}'

    def get_answer(self, completion: str) -> Any:
        completion = completion.split('\n\n')[0]
        return [entry.strip().split(':')[0] for entry in completion.strip().split(self.bullet) if entry.strip()]