# Transtone-Cross-Linguistic-Tone-Controlled-Bilingual-Text-Generation-For-Malayalam-English


## Overview

TransTone is a multilingual NLP framework designed for Malayalam-English translation and tone-controlled text generation. The system combines bilingual machine translation and monolingual style rewriting within a unified architecture.

The framework enables users to generate text in multiple expressive styles while preserving semantic meaning.

Supported tasks:

* English → Malayalam
* Malayalam → English
* English → English
* Malayalam → Malayalam

Supported tones:

* Formal
* Informal
* Poetic

## Architecture

The model is built on Facebook's NLLB-200 (No Language Left Behind) multilingual encoder-decoder architecture.

Parameter-efficient fine-tuning is achieved using LoRA (Low Rank Adaptation) adapters inserted into transformer attention layers.

Input format:

[task:task_name](task:task_name) <tone> source_text

Example:

[task:ml2en](task:ml2en) <formal> സുഖമാണോ?

## Dataset

The project combines:

1. English-Malayalam Parallel Corpus (Kaggle)
2. Custom Tone-Annotated Dataset

Custom dataset statistics:

* 5254 sentence pairs
* Formal: 31.34%
* Informal: 34.33%
* Poetic: 34.33%

## Training Strategy

Stage 1:
General multilingual alignment using English-Malayalam parallel corpus.

Stage 2:
Domain adaptation using custom tone-annotated bilingual dataset.

## Results

| Metric         | Stage 1 | Stage 2 |
| -------------- | ------- | ------- |
| BLEU           | 31.34   | 32.57   |
| chrF           | 54.25   | 44.50   |
| BERTScore (F1) | -       | 0.9431  |

The framework successfully preserves semantic meaning while generating tone-controlled multilingual outputs.

## Technologies

* Python
* PyTorch
* Transformers
* HuggingFace
* NLLB-200
* LoRA
* FastAPI

## Future Work

* Additional Indian language support
* More expressive styles
* Larger tone-annotated datasets
* Speech-to-text integration
* Real-time deployment optimization
