# BabyLM 2026 shared task @ NeTS 

In repository contains the data related to our participation to the [BabyLM 2026](https://babylm.github.io/) competition.

### 01-preprocess
This folder contains our preprocessing procedure. 
We decided to minimally clean the data source by removing any metalinguistic 

### 02-tokenization
You will find here an original tokenizer, dubbed **MorPiece** (**MoP**) freely inspired to the [Tolerance Principle](https://lingbuzz.net/lingbuzz/004146) by Charles Yang.
The current [MorPiece version is v.1.4.5](https://github.com/cristianochesi/morpiece). We compare this with other tokenization approaches.

### 03-model_training
Our original elaboration of some Recurrent Neural Network architectures (**RNNG** and **eMG-RNN** models). These models are inspired by processing interpretations of Minimalist Grammars ([e-MGs](https://github.com/cristianochesi/e-MGs)) and hard-code structural biases (C-command/Merge/Move) and Linguistic parameterization ([Roberts 1999](https://academic.oup.com/book/38726), 
[Chesi 2023](https://doi.org/10.4000/ijcol.1135)).

### 04-evaluation
Results of the evaluation pipeline for BabyLM 2026 are included here.

### REFERENCE

[Chesi et al. 2024 - Different Ways to Forget: Linguistic Gates in Recurrent Neural Networks](https://aclanthology.org/2024.conll-babylm.9/)
