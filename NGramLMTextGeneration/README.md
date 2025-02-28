# N-gram Language Models for Text Generation

This repository contains Python code for building and experimenting with N-gram language models (bigram, trigram, 4-gram, and 5-gram) using the Reuters corpus from the NLTK library. The models are used to predict the next word in a sequence and generate sentences based on the given context.


## Features
- Code to tokenize the Reuters corpus and build N-gram models.
- Functions to predict the next word given a context.
- Functions to generate sentences using the N-gram models.
- Observations on the performance and coherence of the generated sentences for different N-gram models.

## Library Reference
To run the code in this repository, you need to have Python installed along with the following libraries:
- `nltk`: The NLTK library for providing the Reuters corpus and tokenization tools.
- `collections`

## Example Usage
To generate a sentence using the trigram model:

`print(generate_sentence_trigram(trigram_model, 'the', 'stock'))`

To generate a sentence using the 5-gram model:

`print(generate_sentence_fivegram(fivegram_model, 'the', 'stock', 'market', 'collapse', 10))`

## Contributions
Feel free to contribute via pull requests.
