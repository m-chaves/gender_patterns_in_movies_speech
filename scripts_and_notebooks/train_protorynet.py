'''
Created on July 26th 2022
Author: Mariana Chaves

This script trains a protorynet model using as base the code from https://github.com/dathong/ProtoryNet with some modifications. 

Requiered arguments: 
    datasetpath: path to the directory where the train and validation sets can be found. This directory must contain pickle files with the names x_train, y_train, x_val and y_val. 
    results_path: path where we want the results to be saved. It should already exist.
    results_prefix: a string that would be included at the beggining of the name of the pickle file containing the results. 
    epochs: number of epochs to train the model. 
    number_prototypes: number of prototypes in the model.
    type_init: type of initialization for the prototypes. 'kmedoids' or 'random'
    sample_size_sentences = number of sentences to take as samples for the initialization and visualization of prototypes. More tan 30000 makes the kmedoids method run into memory issues. High numbers also make slow the extraction of the initial prototypes. For instance, using the cornell dataset, this could take up to 40min just to extract the initial prototypes.
    init_prototypes_seed: seed for the selection of initial prototypes (in case of choosing 'random' for the prototype initialization)
'''
import time
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
import os
import sys
import myfunctions
# import nltk
# nltk.download('punkt')
sys.path.append('./src/protoryNet/')
from protoryNet import ProtoryNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProtoryNet")
    parser.add_argument("--dataset_path", required = True)
    parser.add_argument("--results_path", required = True)
    parser.add_argument("--results_prefix", required = True)
    parser.add_argument("--epochs", type=int, required = True) 
    parser.add_argument("--number_prototypes", type=int, required = True) 
    parser.add_argument("--type_init", required = False, default = 'random')
    parser.add_argument("--sample_size_sentences", type=int, required = False, default = 20000) 
    parser.add_argument("--init_prototypes_seed", type=int, required = False, default = 16) 
    args = parser.parse_args()

    dataset_path = args.dataset_path
    results_path = args.results_path
    results_prefix = args.results_prefix
    epochs = args.epochs
    number_prototypes = args.number_prototypes
    type_init = args.type_init
    sample_size_sentences = args.sample_size_sentences
    init_prototypes_seed = args.init_prototypes_seed
    
    # Name for results
    results_name = '__'.join([results_prefix, 
                              str(epochs) + 'epochs', 
                              str(number_prototypes) + 'prototypes',
                              type_init + 'type_init',
                              str(sample_size_sentences) + 'sample_size_sentences',
                              str(init_prototypes_seed) + 'init_prototypes_seed'])

    # --------------------------
    # Load data
    # --------------------------
    with open (dataset_path + 'x_train', 'rb') as fp:
        x_train = pickle.load(fp)

    with open (dataset_path + 'y_train', 'rb') as fp:
        y_train = pickle.load(fp)

    with open (dataset_path + 'x_val', 'rb') as fp:
        x_val = pickle.load(fp)

    with open (dataset_path + 'y_val', 'rb') as fp:
        y_val = pickle.load(fp)            


    # --------------------------
    # Data preprocessing
    # --------------------------

    # Guarantee target variable is integer
    y_train = [int(y) for y in y_train]
    y_val = [int(y) for y in y_val]

    # Split text into lists of sentences 
    x_train = myfunctions.split_sentences(x_train)
    x_val = myfunctions.split_sentences(x_val)

    # Make a list of sentences (only for training set)
    train_sentences = []
    for p in x_train:
        train_sentences.extend(p)

    # We remove very short or very long sentences since they behave as outliers.
    train_sentences = [i for i in train_sentences if len(i)>5 and len(i)<100]

    # Import Google Sentence encoder, to convert sentences into vector values
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model_sentence_encoder = hub.load(module_url)
    def embed(input):
        return model_sentence_encoder(input)

    #Compute embeddings of sentences
    train_sentences_embedded = embed(train_sentences)

    # --------------------------
    # Initialize prototypes
    # --------------------------
    initial_prototypes_embedded = myfunctions.initialize_prototypes(type_init, 
                                                        number_prototypes, 
                                                        train_sentences_embedded, 
                                                        sample_size = sample_size_sentences,
                                                        init_prototypes_seed = init_prototypes_seed)

    # --------------------------
    # Train model
    # --------------------------
    # Create model
    pNet = ProtoryNet() 
    # Include the initial prototypes
    model = pNet.createModel(initial_prototypes_embedded, k_protos = number_prototypes)
    
    # Get the initial prototypes as sentences 
    start_time = time.time()
    initial_prototypes = pNet.showPrototypes(train_sentences[0:sample_size_sentences], train_sentences_embedded[0:sample_size_sentences], number_prototypes, printOutput=False, return_prototypes = True)
    print('Initial prototypes:', initial_prototypes)
    execution_time = (time.time() - start_time) / 60
    print('Initial prototypes execution time:', execution_time)
    
    # Train
    start_time = time.time()
    maxEvalRes, history_validation_accuracy = pNet.train(x_train, y_train, x_val, y_val, 
                                                         epochs = epochs, 
                                                         saveModel = True, 
                                                         model_name = results_path + results_name, 
                                                         returnValidationAccuracy = True)
    execution_time = (time.time() - start_time) / 60

    # --------------------------
    # Save results
    # --------------------------
    results = {'train_time': execution_time,
               'history_validation_accuracy': history_validation_accuracy,
               'best_validation_accuracy': maxEvalRes,
               'initial_prototypes': initial_prototypes, 
               'args': args}
    print(results)
    pickle.dump(results, open(results_path + results_name + ".pickle", "wb" ))