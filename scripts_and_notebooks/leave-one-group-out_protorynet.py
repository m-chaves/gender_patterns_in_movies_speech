'''
Author: Mariana Chaves
Date: July 2022

This script performs leave-one-group-out cross-validation on ProtoryNet models. 
Leave-one-group-out is similar to leave-one-out with the difference that instead of leaving just one sample out each time, all samples belonging to a group are lefted out, and the work as test set.  
The dataset should be provided as a csv file that will be read by pandas. 
The dataset must contain a text variable, a binary target variable and a variable that indicates the groups for using leave-one-group-out. 

Requiered arguments: 
    datasetpath: path to the csv file containing the data. 
    results_path: directory where we want the results to be saved. It should already exist.
    results_prefix: a string that would be included at the beggining of the name of the pickle file containing the results. 
    group_variable: the name of the variable that contains the groups (for the leave-one-out)
    text_variable: the name of the variable that contains the text.
    target_variable: the name of the variable that contains the target variable (binary variable).
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
    parser = argparse.ArgumentParser(description="Leve-one-group-out on ProtoryNet")
    parser.add_argument("--dataset_path", required = True)
    parser.add_argument("--results_path", required = True)
    parser.add_argument("--results_prefix", required = True)
    parser.add_argument("--group_variable", required = True)
    parser.add_argument("--text_variable", required = True)
    parser.add_argument("--target_variable", required = True)
    parser.add_argument("--epochs", type=int, required = True) 
    parser.add_argument("--number_prototypes", type=int, required = True) 
    parser.add_argument("--type_init", required = False, default = 'random')
    parser.add_argument("--sample_size_sentences", type=int, required = False, default = 20000) 
    parser.add_argument("--init_prototypes_seed", type=int, required = False, default = 16) 
    args = parser.parse_args()
    
    results_name = '__'.join(['leave-one-group-out',
                              args.results_prefix,
                              args.group_variable + 'groupvariable',
                              str(args.epochs) + 'epochs', 
                              str(args.number_prototypes) + 'prototypes',
                              args.type_init + 'type_init',
                              str(args.sample_size_sentences) + 'sample_size_sentences',
                              str(args.init_prototypes_seed) + 'init_prototypes_seed'])
    
    # Load dataset
    data = pd.read_csv(args.dataset_path)

    # Groups for leave-one-group-out
    groups = np.unique(data[args.group_variable])

    # Empty lists
    execution_time_list = []
    history_validation_accuracy_list = []
    maxEvalRes_list = []
    initial_prototypes_list = []
    accuracy_train_list = []
    accuracy_test_list = []
    preds_test_list = []
    final_prototypes_list = []
    text = []
    ground_truth = []

    # For each group
    for group in groups:


        #-----------------------------
        # Split train val test
        #-----------------------------

        # Note that we need the 3 datasets because in the training process of protorynet, 
        # the model with the best metrics in validation is the one that is saved.
        # In other words, we make a decision using data from validation. 
        # Therefore we need the test set to make the final assessment on accuracy.    

        train_data = data[data[args.group_variable] != group]
        test_data = data[data[args.group_variable] == group]

        x = train_data[args.text_variable]
        y = train_data[args.target_variable]
        x_test = test_data[args.text_variable]
        y_test = test_data[args.target_variable]
        x_train, x_val, y_train, y_val  = myfunctions.balanced_split_train_val_test(x, y, train_split = 0.8, val_split = 0, test_split = 0.2, random_seed = 32)

        # --------------------------
        # Data preprocessing
        # --------------------------

        # Guarantee target variable is integer
        y_train = [int(y) for y in y_train]
        y_val = [int(y) for y in y_val]
        y_test = [int(y) for y in y_test]

        # Split text into lists of sentences 
        x_train = myfunctions.split_sentences(x_train)
        x_val = myfunctions.split_sentences(x_val)
        x_test = myfunctions.split_sentences(x_test)

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
        initial_prototypes_embedded = myfunctions.initialize_prototypes(args.type_init, 
                                                            args.number_prototypes, 
                                                            train_sentences_embedded, 
                                                            sample_size = args.sample_size_sentences,
                                                            init_prototypes_seed = args.init_prototypes_seed)

        # --------------------------
        # Train model
        # --------------------------

        # Create model
        pNet = ProtoryNet() 
        # Include the initial prototypes
        model = pNet.createModel(initial_prototypes_embedded, k_protos = args.number_prototypes)

        # Get the initial prototypes as sentences 
        start_time = time.time()
        initial_prototypes = pNet.showPrototypes(train_sentences[0:args.sample_size_sentences], train_sentences_embedded[0:args.sample_size_sentences], args.number_prototypes, printOutput=False, return_prototypes = True)
        print('Initial prototypes:', initial_prototypes)
        execution_time = (time.time() - start_time) / 60
        print('Initial prototypes execution time:', execution_time)

        # Train
        start_time = time.time()
        maxEvalRes, history_validation_accuracy = pNet.train(x_train, y_train, x_val, y_val, 
                                                             epochs = args.epochs, 
                                                             saveModel = True, 
                                                             model_name = args.results_path + results_name + '__'+ str(group) + 'groupnumber', 
                                                             returnValidationAccuracy = True)
        execution_time = (time.time() - start_time) / 60

        # --------------------------
        # Evaluate model
        # --------------------------
        model_path = args.results_path + results_name + '__'+ str(group) + 'groupnumber' + '.h5'

        # Load model
        pNet_saved = ProtoryNet()
        model = pNet_saved.createModel(np.zeros((args.number_prototypes, 512)), args.number_prototypes)
        model.load_weights(model_path)

        # Sentence embedding using the finetune embedder in the model
        train_sentences_embedded = pNet_saved.embed(train_sentences)

        # Evaluate the model on testing data
        preds_train, accuracy_train = pNet_saved.evaluate(x_train, y_train)
        preds_test, accuracy_test = pNet_saved.evaluate(x_test, y_test)

        # Final_prototypes
        final_prototypes = pNet_saved.showPrototypes(train_sentences, train_sentences_embedded, args.number_prototypes, printOutput=False, return_prototypes = True)

        # --------------------------
        # Save results
        # --------------------------

        execution_time_list.append(execution_time)
        history_validation_accuracy_list.append(history_validation_accuracy)
        maxEvalRes_list.append(maxEvalRes)
        initial_prototypes_list.append(initial_prototypes)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        preds_test_list.extend(preds_test)
        final_prototypes_list.append(final_prototypes)
        text.extend(x_test)
        ground_truth.extend(y_test)

        results = {'train_time': execution_time_list,
                   'history_validation_accuracy': history_validation_accuracy_list,
                   'best_validation_accuracy': maxEvalRes_list,
                   'initial_prototypes': initial_prototypes_list,
                   'predictions_on_test': preds_test_list,
                   'text_test': text,
                   'ground_truth': ground_truth,
                   'train_accuracy': accuracy_train_list,
                   'test_accuracy': accuracy_test_list,
                   'final_prototypes': final_prototypes_list,
                   'args': args}

        pickle.dump(results, open(args.results_path + results_name + ".pickle", "wb" ))

        print('----------------------------')
        print('Finished for group:', group)
        print('----------------------------')


