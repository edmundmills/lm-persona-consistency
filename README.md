# Language Model Persona Consistency

This repo contains the code associated with this [blog post](https://edmundmills.com/blog/storytelling-makes-gpt-35-deontologist-unexpected-effects-of-context-on-llm-behavior).

## Overview

When prompted to make decisions, do large language models (LLMs) show power-seeking behavior, self-preservation instincts, and long-term goals? Discovering Language Model Behaviors with Model-Written Evaluations (Perez et al.) introduced a set of evaluations for these behaviors, along with other dimensions of LLM self-identity, personality, views, and decision-making. Ideally, we’d be able to use these evaluations to understand and make robust predictions about safety-relevant LLM behavior. However, these evaluations invite the question: is the measured behavior a general property of the language model, or is it closely tied to the particular context provided to the language model?

In this work, we measure the consistency of LLM behavior over a variety of ordinary dialogue contexts. We find that with existing language models, the robustness of a given behavior can vary substantially across different tasks. For example, asking GPT-3.5 (text-davinci-003) to write stories tends to make it subscribe more to Deontology. Viewing these results in the simulator framework, we see this as a shift in the persona that the model is simulating. Overall, our work indicates that care must be taken when using a question-answer methodology to evaluate LLM behavior. Results from benchmarks such as Perez et al. might not generalize to dialogue contexts encountered in the wild.

## Code Structure

### Requirements

- Python 3.8

### Setup

1. Install the package with `pip install -e .`
2. Set your OPENAI_API_KEY with `export OPENAI_API_KEY=<key>`

### Running the Experiments

Please note that this was a very quick project. As such, to run experiments with different models, tasks, and persona-inducing contexts, change the variables in the provided scripts.

#### Generating Dialogue Contexts

`python persona_inducers.py`

#### Running the Evaluation

`python evaluate.py`

#### Plotting

`python make_plots.py`
