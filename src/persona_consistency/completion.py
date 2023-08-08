import logging
import math
import time
from abc import ABC, abstractmethod
from collections import abc
from copy import deepcopy
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openai
from tqdm import tqdm

from persona_consistency.prompting import Prompter
from persona_consistency.questions import (ExactMatchQuestion,
                                           FreeResponseQuestion,
                                           MultipleChoiceQuestion, Question)

gpt_letter_tokens = [317, 347, 327, 360, 412, 376, 402, 367, 314, 449, 509, 406, 337, 399, 440, 350, 1195, 371, 311, 309, 471, 569, 370, 1395, 575, 1168]  # " A", " B", etc.

GPT_MC_TOKENS = {l: t for l, t in zip(ascii_uppercase, gpt_letter_tokens)}
GPT_MC_TOKENS['Yes'] = 3363
GPT_MC_TOKENS['No'] = 1400

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    """
    Represents a completion response from a model.

    Attributes:
        prompt (str): The prompt text that was sent to the model.
        text (str): The completion text that was returned by the model.
        meta (Optional[Dict]): Additional metadata associated with the completion.
        logprobs (Optional[List[Dict[str, float]]]): Log probabilities associated with the completion.
    """
        
    prompt: str
    text: str
    meta: Optional[Dict] = None
    logprobs: Optional[List[Dict[str, float]]] = None

    def __repr__(self):
        return f'{self.__class__.__name__}: Prompt: {self.prompt}, Completion Text: {self.text}'

    @property
    def mc_answer_probs(self):
        assert self.logprobs is not None
        return {token.strip(): math.exp(logprob) for token, logprob in self.logprobs[0].items()}

    def to_dict(self, include_logprobs: bool = False) -> Dict[str, Any]:
        ret: Dict[str, Any] = dict(prompt=self.prompt, completion=self.text, meta=self.meta)
        if self.logprobs and len(self.logprobs) == 1:
            ret['answer_probs'] = self.mc_answer_probs
        if include_logprobs:
            ret['logprobs'] = self.logprobs
        return ret


class Completer(ABC):
    """
    Abstract base class for completers. Defines the interface that all completers must implement.
    """
    
    @abstractmethod
    def __call__(self, prompts: List[str], *args, **kwargs) -> List[Completion]:
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ThrottledCompleter(Completer):
    """
    A wrapper around a completer that adds throttling functionality.

    Attributes:
        completer (Completer): The underlying completer.
        iteration_pause_seconds (int): Number of seconds to pause between iterations.
        retry_pause_seconds (int): Number of seconds to pause between retries.
        max_tries (int): Maximum number of retry attempts.
        last_called_time (int): Timestamp of the last call to the completer.
        completion_batch_size (int): Size of the batch for completion.
    """
    
    def __init__(self, completer: Completer, pause_seconds: int = 20, completion_batch_size: int = 1, retry_pause_seconds: int = 30) -> None:
        self.completer = completer
        self.iteration_pause_seconds = pause_seconds
        self.retry_pause_seconds = retry_pause_seconds
        self.max_tries = 5
        self.last_called_time = 0
        self.completion_batch_size = completion_batch_size

    def __repr__(self) -> str:
        return str(self.completer)

    @property
    def name(self):
        return self.completer.name

    def throttled_complete(self, prompts: List[str], *args, **kwargs) -> List[Completion]:
        tries = 0
        completions = []
        wait_for_seconds = self.last_called_time + self.iteration_pause_seconds - time.time()
        while not completions and tries < self.max_tries:
            if wait_for_seconds > 0:
                time.sleep(wait_for_seconds)
            try:
                completions = self.completer(prompts, *args, **kwargs)
            except Exception as e:
                print(e)
                print(f'Waiting for {self.retry_pause_seconds} seconds')
                tries += 1
                wait_for_seconds = self.retry_pause_seconds
                completions = []
        self.last_called_time = time.time()
        return completions

    def __call__(self, prompts: List[str], *args, **kwargs) -> List[Completion]:
        n_batches = math.ceil(len(prompts) / self.completion_batch_size)
        completions = []
        for i in tqdm(range(n_batches), total=n_batches, desc='Generating Completions'):
            prompt_batch = prompts[i*self.completion_batch_size:(i+1)*self.completion_batch_size]
            completion_batch = self.throttled_complete(prompt_batch, *args, **kwargs)
            completions.extend(completion_batch)
            for p, c in zip(prompt_batch, completion_batch):
                logging.debug(f'---\n{p}\n\n{c.text}')
        return completions


class GPT3(Completer):
    """
    A completer that uses the GPT-3 model.

    Attributes:
        completion_params (Dict[str, Any]): Parameters for the completion request.
    """

    def __init__(self, model_name='text-davinci-003', temperature: float = 0.0, max_tokens: int = 1024) -> None:
        self.completion_params: Dict[str, Any] = dict(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,
        )

    @property
    def name(self):
        return self.completion_params['model']

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}: ({str(self.completion_params)})'

    def __call__(self, prompts: List[str], options: Optional[List[str]] = None) -> List[Completion]:
        completion_params = deepcopy(self.completion_params)
        if options is not None:
            completion_params['logit_bias'] = {GPT_MC_TOKENS[o]: 100 for o in options}
            completion_params['logprobs'] = len(options)
            completion_params['max_tokens'] = 1
        response = openai.Completion.create(
            prompt=prompts,
            **completion_params
        )
        return [
            Completion(
                prompt,
                choice['text'],
                dict(
                    completion_params=completion_params,
                    finish_reason=choice['finish_reason']
                ),
                logprobs=[dict(lp) for lp in choice['logprobs']['top_logprobs']]
            ) for prompt, choice in zip(prompts, response['choices'])  # type: ignore
        ]


@dataclass
class Answer:
    """
    Represents an answer to a question.

    Attributes:
        question (Question): The question being answered.
        completion (Completion): The completion used to generate the answer.
        prompter (Prompter): The prompter used to generate the prompt.
    """

    question: Question
    completion: Completion
    prompter: Prompter

    @property
    def given_answer(self) -> Any:
        return self.prompter.get_answer(self.completion.text)

    @property
    def is_correct(self) -> bool:
        return self.question.is_correct(self.given_answer)

    def to_dict(self):
        data = self.completion.to_dict()
        if isinstance(self.question, (MultipleChoiceQuestion, ExactMatchQuestion)):
            data['label'] = self.question.label
        return data


@dataclass
class AnswerSet(abc.Sequence):
    """
    Represents a set of answers.

    Attributes:
        answers (List[Answer]): List of answers.
    """
    
    answers: List[Answer]

    @property
    def prompter(self):
        return self.answers[0].prompter

    @property
    def questions(self):
        return [a.question for a in self.answers]

    @property
    def completions(self):
        return [a.completion for a in self.answers]

    def __repr__(self):
        return f"{self.__class__.__name__}: {dict(example_question=self.questions[0], example_completion=self.completions[0], prompter=self.prompter)}"

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx) -> Answer:
        return self.answers[idx]

    @property
    def given_answers(self) -> List[Any]:
        return [a.given_answer for a in self.answers]

    @property
    def mc_answer_probs(self) -> List[Dict[str, float]]:
        return [c.mc_answer_probs for c in self.completions]

    @property
    def correct(self) -> List[bool]:
        return [a.is_correct for a in self.answers]

    @property
    def accuracy(self) -> float:
        return np.array(self.correct, dtype=float).mean()

    @property
    def mean(self) -> float:
        assert all(isinstance(a, float) for a in self.given_answers)
        return np.array(self.given_answers).mean()

    @property
    def earliest_true_index(self) -> Union[int, None]:
        for i, correct in enumerate(self.correct):
            if correct:
                return i
        return None

    def to_dict(self):
        return dict(answers=[a.to_dict() for a in self])


def answer(questions: List[Union[Question, FreeResponseQuestion, MultipleChoiceQuestion]], prompter: Prompter, completer: Completer) -> AnswerSet:
    """
    Answers a list of questions using the given prompter and completer.

    Args:
        questions (List[Union[Question, FreeResponseQuestion, MultipleChoiceQuestion]]): List of questions to answer.
        prompter (Prompter): The prompter used to generate the prompts.
        completer (Completer): The completer used to generate the completions.

    Returns:
        AnswerSet: A set of answers to the questions.
    """

    if not questions:
        return AnswerSet([])
    prompts = [prompter.make_prompt(q) for q in questions]
    completions = completer(prompts)
    answers = [Answer(q, c, prompter) for q, c in zip(questions, completions)]
    return AnswerSet(answers)


def answer_multiple_choice(questions: List[MultipleChoiceQuestion], prompter: Prompter, completer: Completer) -> AnswerSet:
    """
    Answers a list of multiple-choice questions using the given prompter and completer.

    Args:
        questions (List[MultipleChoiceQuestion]): List of multiple-choice questions to answer.
        prompter (Prompter): The prompter used to generate the prompts.
        completer (Completer): The completer used to generate the completions.

    Returns:
        AnswerSet: A set of answers to the questions.
    """

    if not questions:
        return AnswerSet([])
    options = list(questions[0].options.keys())
    assert all(list(q.options.keys()) == options for q in questions)
    prompts = [prompter.make_prompt(q) for q in questions]
    completions = completer(prompts, options)
    answers = [Answer(q, c, prompter) for q, c in zip(questions, completions)]
    return AnswerSet(answers)