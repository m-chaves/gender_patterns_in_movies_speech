'''
Created on 21 jun. 2022
Author: Mariana Chaves

This script trains a word2vec model using gensim's implementation. 
It was created taking into consideration the use of leave-one-GROUP-out crossvalidation. 
That is, we can specify a group that we want to remove from our training set. 
The dataset that we provide as input should be a pandas dataframe saved using pickle. 
The pandas should include a text variable 
If you intend to use leave-one-GROUP-out crossvalidation, a variable based on the group should be included in dataframe. 

Requiered arguments: 
- input: a pandas dataframe (saved using pickle or as a csv) containing the text variable, and an optional variable defining the groups. 
- output: the path to save the word2vec model, including the name for it. Ex: results/trial, then the model will be saved as trial.wordvectors  
- text_variable: name of the variable that contains the text in the pandas dataframe
Optional arguments:
- group_variable: name of the variable that contains the groups in the pandas dataframe
- group_index: an integer indicating the group that will be excluded. 0 indicates the first group, 1 the second one and so on.
- embedding_size: an integer indicating the size of the embedding. 100 by default.
- window_size: an integer indicating the maximum distance between the current and predicted word within a sentence. 5 by default.
- min_count:  ignores all words with total frequency lower than this. 5 by default.
- sg: 0 or 1. 0 = continuous bag of words, 1 = skip-gram. Skip-gram by default.
- lower: boolean. If True, puts everything in lower case. 
'''
import sys
import os
import json
import gensim
import argparse
import pandas as pd
import pickle

if __name__ == '__main__':

    # Log parameters for easier identification of logs
    parser = argparse.ArgumentParser(description="Train word2vec")
    parser.add_argument("--input", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--text_variable", required = True)
    parser.add_argument("--group_variable", required = False)
    parser.add_argument("--group_index", type=int, required = False) 
    parser.add_argument("--embedding_size", type=int, required = False) 
    parser.add_argument("--window_size", type=int, required = False) 
    parser.add_argument("--min_count", type=int, required = False) 
    parser.add_argument("--sg", type=int, required = False) 
    parser.add_argument("--lower", type=int, required = False, default=True) 
    args = parser.parse_args()
    
    # Load pandas dataset
    print("-"*50)
    print('Loading dataset')
    if args.input[-3:] == 'pkl':
      dataset = pd.read_pickle(args.input)
    elif  args.input[-3:] == 'csv':
      dataset = pd.read_csv(args.input)
    else:
      raise ValueError('dataset should be a pickle or csv file')

    # Leave one group out (if necessary)
    # The group to remove will be determined by the group_index argument
    if args.group_index != None:
      print("-"*50)
      print('Using leave-one-group-out')
      print('Excluding group number', args.group_index)
      codes, uniques = pd.factorize(dataset[args.group_variable], sort = False)
      dataset['group_index'] = codes
      dataset = dataset[dataset['group_index'] != args.group_index]
    args.group_index

    # Put everything in lower case
    if args.lower:
      dataset[args.text_variable]= [x.lower() for x in dataset[args.text_variable]]

     # Keep only the text variable
    text_variable = dataset[args.text_variable]

    # Create a temporal text file. Each dialogue its a line.
    text = '\n'.join(dataset[args.text_variable])
    text_tmp_file = "temporal_text.tmp"
    f = open(text_tmp_file, "w")
    f.write(text)
    f.flush()
    f.close()

    # Use gensim's way of efficiently reading text        
    sentences = gensim.models.word2vec.LineSentence(text_tmp_file)

    # Model parameters
    # Use default parameters unless parameters are specified 
    # By default the embedding is of size 100
    vector_size = args.embedding_size if args.embedding_size != None else 100
  	# By default the window is of size 5
    window_size = args.window_size if args.window_size != None else 5
  	# By default words that appear less than 5 times are not taken into consideration
    min_count = args.min_count if args.min_count != None else 5
    # By default skip-gram is used. Use 0 to use continuous bag of words. 
    sg = args.sg if args.min_count == 0 else 1

    # Train word2vec
    print("-"*50)
    print('Training word2vec')
    model = gensim.models.Word2Vec(sentences=sentences, size = vector_size, window = window_size, min_count = min_count, sg=args.sg, workers=4)

    # Save model
    word_vectors = model.wv
    word_vectors.save(args.output + ".wordvectors")
    print("-"*50)
    print('Word2vec done')

    # Delete temporal file
    os.system('rm -rf ' + text_tmp_file + "*" )

            
    