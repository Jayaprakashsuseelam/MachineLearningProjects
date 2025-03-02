# Neural Machine Translation with Attention Mechanism

This repository contains the implementation of a Neural Machine Translation (NMT) model using an Encoder-Decoder architecture with and without Attention Mechanism. The model is trained on a dataset containing English-Hindi sentence pairs. The implementation is done using both PyTorch and TensorFlow.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Introduction
Neural Machine Translation (NMT) is a approach to machine translation that uses a large neural network to predict the likelihood of a sequence of words, typically in a target language given a sequence of words in a source language. This project implements an NMT model using an Encoder-Decoder architecture, with and without an Attention Mechanism, to translate sentences from English to Hindi.

## Features
- **Encoder-Decoder Architecture**: The model uses an LSTM-based Encoder-Decoder architecture for sequence-to-sequence learning.
- **Attention Mechanism**: Implements Bahdanau Attention to improve translation quality by focusing on relevant parts of the input sequence.
- **BLEU Score Evaluation**: The model's performance is evaluated using the BLEU score metric.
- **PyTorch and TensorFlow Implementations**: The project includes implementations in both PyTorch and TensorFlow.

## Installation
To run this project, you need to have Python 3.7 or higher installed. Follow these steps to set up the environment:
