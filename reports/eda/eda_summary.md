# RAGTruth Exploratory Data Analysis

## Dataset Overview

- Total responses: 17,790
- Supported: 10,126
- Hallucinated: 7,664
- Overall hallucination rate: 43.08%
- Unique sources: 2,965
- Generation models: 6
- Task types: 3

## Questions Explored

1. Class balance
2. Split composition
3. Answer-length differences
4. Context-length distribution
5. Model-level hallucination rates
6. Hallucination rate across answer-length deciles
7. Lexical grounding differences
8. Hallucination annotation taxonomy
9. Temperature relationship
10. Task-type relationship

## Model-Level Rates

- mistral-7B-instruct: 65.87%
- llama-2-7b-chat: 61.79%
- llama-2-13b-chat: 56.56%
- llama-2-70b-chat: 47.05%
- gpt-4-0613: 13.69%
- gpt-3.5-turbo-0613: 13.52%

## Task-Type Rates

- Data2txt: 68.64%
- Summary: 29.80%
- QA: 29.05%

## Annotation Types

- Evident Baseless Info: 6,237
- Evident Conflict: 5,324
- Subtle Baseless Info: 2,527
- Subtle Conflict: 201

## Numerical Summary

| group        |   records |   answer_words_mean |   answer_words_median |   context_words_mean |   context_words_median |   lexical_overlap_mean |   annotation_density_mean |
|:-------------|----------:|--------------------:|----------------------:|---------------------:|-----------------------:|-----------------------:|--------------------------:|
| Supported    |     10126 |             118.785 |                   111 |              382.594 |                    319 |               0.683331 |                 0         |
| Hallucinated |      7664 |             146.231 |                   138 |              378.295 |                    319 |               0.512959 |                 0.0135977 |

## Interpretation Note

These are descriptive dataset-level observations. They do not establish causal relationships. In particular, model, temperature, task type, answer length, and hallucination behavior can be confounded by how the original dataset was generated.