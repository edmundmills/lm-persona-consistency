import json
import random
from pathlib import Path

from persona_consistency.completion import GPT3, ThrottledCompleter, answer
from persona_consistency.datasets import load_simulation_inducing_questions
from persona_consistency.prompting import FreeResponsePrompter
from persona_consistency.questions import FreeResponseQuestion


def main():
    model_name = 'text-davinci-003'
    completer = ThrottledCompleter(GPT3(model_name, max_tokens=300, temperature=0.7), completion_batch_size=20)

    # random.seed(0)
    # n_personas = 20
    # simulation_questions = load_simulation_inducing_questions(Path('questions/simulation_inducing_questions.json'))
    # simulation_questions = random.sample(simulation_questions, n_personas)
    
    simulation_questions = [
        FreeResponseQuestion('Write a poem in the first person:'),
        FreeResponseQuestion('Write a brief short story in the first person:'),
        FreeResponseQuestion('Write a short satirical piece on the topic of your choice:'),
        FreeResponseQuestion('Write a short opinion piece on the topic of your choice:'),
    ]
    simulation_questions = [item for item in simulation_questions for i in range(4)]

    simulation_answers = answer(simulation_questions, FreeResponsePrompter(), completer)  # type: ignore
    with open(f'persona_inducers/{model_name}.json', 'w') as f:
        json.dump(simulation_answers.to_dict(), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()