# English–Malayalam Parallel Corpus

## Overview

This dataset is used during Stage 1 training of the TransTone framework.

The corpus contains parallel English–Malayalam sentence pairs and is used to establish multilingual translation alignment between English and Malayalam.

The dataset serves as the foundation for bilingual machine translation before tone adaptation is introduced.

## Source

Kaggle English–Malayalam Parallel Corpus

Reference:
“English–Malayalam Parallel Corpus.”
https:
//www.kaggle.com/datasets/subinek/english-malayalam-parallel-corpus

## Purpose

The dataset is used for:

* English → Malayalam Translation
* Malayalam → English Translation

Stage 1 training focuses on learning semantic alignment between the two languages.

## Dataset Structure

| Column           | Description         |
| ---------------- | ------------------- |
| source_malayalam | Malayalam sentence  |
| target_english   | English translation |

Example:

| source_malayalam         | target_english |
| ------------------------ | -------------- |
| സുഖമാണോ?                 | How are you?   |
| നന്ദി                    | Thank you      |
| എനിക്ക് പുസ്തകം ഇഷ്ടമാണ് | I like books   |

## Preprocessing

The following preprocessing steps are applied:

1. Remove null values
2. Remove duplicate sentence pairs
3. Strip whitespace
4. Remove empty rows
5. Length-based stratification for train-validation split

## Train-Validation Split

Training Set: 90%

Validation Set: 10%

The split is performed using sentence-length stratification to maintain balanced sequence distributions.

## Role in TransTone

This dataset is used during:

Stage 1 Training

Goal:

Learn Malayalam-English translation mappings and semantic alignment using the NLLB-200 multilingual model.

The resulting model serves as the initialization for Stage 2 tone-aware adaptation.
# TransTone Custom Tone-Annotated Malayalam–English Dataset

## Overview

The TransTone Custom Dataset is a bilingual Malayalam–English dataset created for tone-controlled multilingual text generation.

The dataset contains sentence pairs annotated with stylistic labels to support controlled text generation while preserving semantic meaning.

The dataset was developed as part of the TransTone research project.

## Dataset Statistics

Total Sentence Pairs: 5,254

Tone Distribution:

| Tone     | Percentage |
| -------- | ---------- |
| Formal   | 31.34%     |
| Informal | 34.33%     |
| Poetic   | 34.33%     |

The dataset maintains a balanced distribution across all tone categories.

## Supported Tasks

The dataset supports four tasks:

### English → Malayalam (en2ml)

Translation from English to Malayalam while preserving tone.

Example:

Source:
Life is beautiful.

Target:
ജീവിതം മനോഹരമാണ്.

Tone:
Informal

### Malayalam → English (ml2en)

Translation from Malayalam to English while preserving tone.

Example:

Source:
വിദ്യാഭ്യാസം ഏറ്റവും വലിയ സമ്പത്താണ്.

Target:
Education is the greatest wealth of a person.

Tone:
Formal

### English → English (en2en)

Monolingual style rewriting.

Example:

Source:
Please help me.

Target:
I kindly request your assistance.

Tone:
Formal

### Malayalam → Malayalam (ml2ml)

Monolingual Malayalam style transformation.

Example:

Source:
സഹായിക്കൂ.

Target:
ദയവായി എന്നെ സഹായിക്കാമോ?

Tone:
Formal

## Dataset Format

| Column      | Description                |
| ----------- | -------------------------- |
| source_text | Input sentence             |
| target_text | Output sentence            |
| tone        | Formal / Informal / Poetic |

Example:

| source_text                       | tone     | target_text                        |
| --------------------------------- | -------- | ---------------------------------- |
| The sun rises every day.          | Formal   | സൂര്യൻ എല്ലാ ദിവസവും ഉദിക്കുന്നു.  |
| Life is beautiful.                | Informal | ജീവിതം മനോഹരമാണ്.                  |
| Dreams are the wings of the soul. | Poetic   | സ്വപ്നങ്ങൾ ആത്മാവിന്റെ ചിറകുകളാണ്. |

## Annotation Strategy

Each sentence pair is manually assigned a stylistic label:

* Formal
* Informal
* Poetic

Annotations are designed to preserve:

* Semantic meaning
* Cultural relevance
* Linguistic naturalness
* Tone consistency

## Preprocessing

The following preprocessing operations are applied:

1. Text normalization
2. Duplicate removal
3. Invalid sample removal
4. Tone validation
5. Automatic task detection
6. Dataset balancing
7. Train-test splitting

## Input Format Used During Training

The model receives task and tone conditioning tokens.

Example:

[task:ml2en](task:ml2en) <formal> സുഖമാണോ?

[task:en2ml](task:en2ml) <poetic> The stars whisper in silence.

## Role in TransTone

This dataset is used during:

Stage 2 Training

Goal:

Enable tone-controlled bilingual translation and monolingual rewriting using explicit task and style conditioning.

The dataset allows the model to generate:

* Formal text
* Informal text
* Poetic text

while preserving the original meaning.

## Research Contribution

This dataset is one of the primary contributions of the TransTone framework.

It enables:

* Tone-aware bilingual generation
* Style-controlled translation
* Multilingual text rewriting
* Malayalam-English expressive language generation

## License

Research and educational use only.





