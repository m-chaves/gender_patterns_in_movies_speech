This repository uses de [Cornell dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) to explore the capabilities of different models on the task of classifying characters on male or female based on their dialogues.

See the full written report [here](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/report.pdf)

We first combine the cornell corpus information with information of [tmdb](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and our manual gender anotations. 
Subsequently we perform some data preprocessing steps. See the [cornell_corpus_preprocessing notebook](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/cornell_corpus_preprocessing.ipynb) to get all the details of this step. 

We experiment with different classification models. 
* Naive bayes and logistic regression models can be found [here](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/sklearn_models.ipynb)
* Deep models (LTSMs, Bidirectional LSTMs and dense networks) can be found in two notebooks:
    - [deep_models_explained.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/deep_models_explained.ipynb) explains in more detail the training of the models and their architecture. This notebook has mostly didactical pursoses. 
    - [deep_models.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/deep_models.ipynb), on the other hand, contains a more compact version for training many different models. This notebook contains the results documented in the written report related to deep models.   
* BERT models can be found in two models:
    - [BERT_explained.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/BERT_explained.ipynb), similarly to deep_models_explained.ipynb, explains in more detail the training for BERT models. This notebook has mostly didactical pursoses. 
    - [BERT_models.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/BERT_models.ipynb) contains a more compact version of the code. This notebook contains the results documented in the written report related to BERT models.   
* Protorynet models can be found in the following notebooks:
    - [challenging_the_interpretability_of_protorynet.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/challenging_the_interpretability_of_protorynet.ipynb) reproduces the example presented by the authors of ProtoryNet in their paper and their github repository, where they classify positive and negative reviews. We show that after a couple of epochs the model achieves 93.56% of accuracy but the prototypes result uninformative. That is, the provide little explanaitions.
    -[protorynet_prototype_initializations.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/protorynet_prototype_initializations.ipynb) compares two approaches for prototypes initialization.  
    - [protorynet_models_on_cornell.ipynb](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/protorynet_models_on_cornell.ipynb) contains the results of ProtoryNet models trained on the Cornell corpus. The scripts use for model training are [train_protorynet.py](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/train_protorynet.py) and [leave-one-group-out_protorynet.py](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/leave-one-group-out_protorynet.py).   

We also train a word2vec model from scratch using the cornell corpus. Take a look at [word2vec.ipynb ](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/word2vec.ipynb) and [train_word2vec.py](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/train_word2vec.py) 

We evaluate possible human biases absorbed by the word2vec model using the Word-Embedding Association Test (WEAT). Take a look at [WEAT.py](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/WEAT.py) and [WEAT_output.txt](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/WEAT_output.txt)

[myfunctions.py](https://gitlab.inria.fr/anr-tractive/wp2/self-explainable-models/text-models/-/blob/main/scripts_and_notebooks/myfunctions.py) is a module with functions that we repeatedly use across many of the notebooks and python scripts. 

Note: The src folder contains the source code for protorynet models. This code is based on the [repository of the autors of protorynet](https://github.com/dathong/ProtoryNet). We include some slight modifications and improvements.  

