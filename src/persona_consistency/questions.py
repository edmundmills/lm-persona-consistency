from abc import ABC, abstractmethod
from string import ascii_uppercase
from typing import Any, Callable, Mapping, Optional, Union


class Question(ABC):
    """
    Abstract base class for representing a question.

    Attributes:
        text (str): The text of the question.

    Methods:
        is_correct: Abstract method to determine if a given answer is correct.
    """

    text: str

    @abstractmethod
    def is_correct(self, answer: Any) -> bool:
        """Determine if the given answer is correct for this question.

        Args:
            answer (Any): The answer to check.

        Returns:
            bool: True if the answer is correct, False otherwise.
        """

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.text}'


class MultipleChoiceQuestion(Question):
    """
    Class for representing a multiple-choice question.

    Attributes:
        option_chars (str): Characters representing the options.
        question_text (str): The text of the question.
        options (Mapping[str, str]): The options for the question.
        label (Optional[str]): The correct answer label.
    """

    option_chars = ascii_uppercase

    def __init__(self, text: str, options: Mapping[str, str], label: Optional[str] = None) -> None:
        if label is not None and label not in options:
            raise ValueError('Label must be in options')
        self.question_text = text
        self.options = options
        self.label = label

    @property
    def text(self):
        """Return the full text of the question, including options."""
        return self.question_text + '\n' + '\n'.join(f'{c}: {a}' for c, a in self.options.items())

    def is_correct(self, answer: str) -> bool:
        """Determine if the given answer is correct for this multiple-choice question."""
        if self.label is None:
            raise AttributeError('Label is None')
        return answer.lower() == self.label.lower().strip()


class YesNoQuestion(MultipleChoiceQuestion):
    """
    Class for representing a yes/no question.

    Attributes:
        option_chars (List[str]): List containing 'Yes' and 'No'.
        question_text (str): The text of the question.
        options (Mapping[str, str]): The options for the question.
        label (Optional[str]): The correct answer label.
    """

    option_chars = ['Yes', 'No']

    def __init__(self, text: str, label: Optional[str] = None) -> None:
        if label is not None and label not in self.option_chars:
            raise ValueError('Label must be in options')
        self.question_text = text
        self.options = {o: o for o in self.option_chars}
        self.label = label

    @property
    def text(self):
        return self.question_text


class ScalarQuestion(Question):
    """
    Class for representing a scalar question.

    Attributes:
        text (str): The text of the question.
        label (Optional[float]): The correct answer label.
        is_correct_fn (Optional[Callable]): Function to determine if an answer is correct.
    """

    def __init__(
            self,
            text: str,
            label: Optional[float] = None,
            is_correct_fn: Optional[Callable[[Union[float, None], float], bool]] = None
            ) -> None:
        self.text = text
        self.label = label
        self.is_correct_fn = is_correct_fn

    def is_correct(self, answer: float) -> bool:
        assert self.label is not None and self.is_correct_fn is not None
        return self.is_correct_fn(answer, self.label)


class ExactMatchQuestion(Question):
    """
    Class for representing an exact match question.

    Attributes:
        text (str): The text of the question.
        label (str): The correct answer label.
    """

    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label = label

    def is_correct(self, answer: str) -> bool:
        assert self.label is not None
        return answer == self.label


class FreeResponseQuestion(Question):
    """
    Class for representing a free-response question.

    Attributes:
        text (str): The text of the question.
        reference_answer (Optional[str]): A reference answer for the question.
        is_correct_fn (Optional[Callable]): Function to determine if an answer is correct.
    """

    def __init__(
            self,
            text: str,
            reference_answer: Optional[str] = None,
            is_correct_fn: Optional[Callable[[str, Optional[str]], bool]] = None
            ) -> None:
        self.text = text
        self.reference_answer = reference_answer
        self.is_correct_fn = is_correct_fn

    def is_correct(self, answer: str) -> bool:
        if self.is_correct_fn is None:
            return True
        return self.is_correct_fn(answer, self.reference_answer)