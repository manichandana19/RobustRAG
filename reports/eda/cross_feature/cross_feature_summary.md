# Cross-Feature EDA

## Derived Features

- Answer/context length ratio
- Answer/context length difference

## Source-Level Agreement

- Sources with all 6 models supported: 269
- Sources with all 6 models hallucinating: 86

## Pairwise Model Disagreement

| model_a            | model_b             |   disagreement_rate |
|:-------------------|:--------------------|--------------------:|
| gpt-3.5-turbo-0613 | gpt-4-0613          |            0.188533 |
| gpt-3.5-turbo-0613 | llama-2-13b-chat    |            0.466779 |
| gpt-3.5-turbo-0613 | llama-2-70b-chat    |            0.404722 |
| gpt-3.5-turbo-0613 | llama-2-7b-chat     |            0.525126 |
| gpt-3.5-turbo-0613 | mistral-7B-instruct |            0.558516 |
| gpt-4-0613         | llama-2-13b-chat    |            0.466442 |
| gpt-4-0613         | llama-2-70b-chat    |            0.412479 |
| gpt-4-0613         | llama-2-7b-chat     |            0.536931 |
| gpt-4-0613         | mistral-7B-instruct |            0.561551 |
| llama-2-13b-chat   | llama-2-70b-chat    |            0.290051 |
| llama-2-13b-chat   | llama-2-7b-chat     |            0.340304 |
| llama-2-13b-chat   | mistral-7B-instruct |            0.353457 |
| llama-2-70b-chat   | llama-2-7b-chat     |            0.355818 |
| llama-2-70b-chat   | mistral-7B-instruct |            0.381113 |
| llama-2-7b-chat    | mistral-7B-instruct |            0.375379 |

## Important Interpretation

These analyses are descriptive. Interactions can reveal dataset structure, but they do not establish causal effects.
Model and temperature are especially likely to be confounded by the original generation setup.