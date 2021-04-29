# DISP

## Introduction

This repo cloned from repo https://github.com/joey1993/bert-defenderand and updated for purpose of the college project only.

## Requirements
```
torch==1.8.1+cu111 
torchvision==0.9.1+cu111 
torchaudio==0.8.1
boto3
hnswlib==0.5.1
nltk
pytest
sklearn
pandas
CUDA 10.0+
gensim==3.8.1
python-Levenshtein
allennlp
dynet --cd to the project root folder and install 
tqdm
```

## Pre-training Discriminator

We first attack the training data on word level or character level. Then we pre-train a discriminator with the adversarial data.

~~~bash
python run/run_bert_discriminator_train.py
~~~

## Pre-training Embedding Estimator

We build a pre-training dataset for embedding estimator by collecting the context of window size for each word in the dataset. It can also be considered as fine-tuning a bert language model using a smaller corpus. The embedding estimator is different from a language model because it only estimate the embedding for a masked token instead of using a huge softmax to pinpoint the word.

```bash
python run/run_bert_generator_train.py
```

## Generate Attacks

We first attack the test data using 4 differernt methods to drop the model performance as much as possible. 

```bash
python run/run_bert_enumerate_attacks_add.py
python run/run_bert_enumerate_attacks_drop.py
python run/run_bert_enumerate_attacks_random.py
python run/run_bert_enumerate_attacks_swap.py
```

## Evaluate Discriminator on all 4 attacks

During this phase, we use the pre-trained discriminator to identify the words that have been attacked.

```bash
python run/run_bert_discriminator_eval_add.py
python run/run_bert_discriminator_eval_drop.py
python run/run_bert_discriminator_eval_random.py
python run/run_bert_discriminator_eval_swap.py
```

## Evaluate Embedding Estimator on all 4 attacks

Then, we recover the words with a pre-trained embedding estimator. Note that we use small-world-graph to conduct a KNN-based search for closest word in the embedding space. 

```bash
python run/run_bert_generator_eval_add.py
python run/run_bert_generator_eval_drop.py
python run/run_bert_generator_eval_random.py
python run/run_bert_generator_eval_swap.py
```

## Pre-Train BERT Classifier

We Pre-train BERT CLassifier before evaluation.

```bash
python run/run_bert_classifier_train.py
```

## Evaluate BERT Classifier in attack free data

We evaluate the performance of BERT on attack free data.

```bash
python run/run_bert_classifier_eval.py
```

## Evaluate BERT classifier on all 4 attacks

We evaluate the performance of BERT on perturbed data.

```bash
python run/run_bert_classifier_eval_add.py
python run/run_bert_classifier_eval_drop.py
python run/run_bert_classifier_eval_random.py
python run/run_bert_classifier_eval_swap.py
```

## Evaluate BERT classifier on all 4 attacks after recovered texts

```bash
python run/run_bert_classifier_eval_add_recovered.py
python run/run_bert_classifier_eval_drop_recovered.py
python run/run_bert_classifier_eval_rand_recovered.py
python run/run_bert_classifier_eval_swap_recovered.py
```

## Pre-Train Discriminator if using GAN2vec and Rob-GAN

```bash
python run/run_GAN2vec_RobGAN_train.py
```
