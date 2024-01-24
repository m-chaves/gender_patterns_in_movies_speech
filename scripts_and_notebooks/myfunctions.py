import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_extra.cluster import KMedoids
import pickle
import re
import gensim
import nltk
nltk.download('punkt')

def balanced_split_train_val_test(X, y, train_split, val_split, test_split, random_seed = 32):
    '''
    Takes features X and the categorical target variable y and split in train, validation and test. 
    It stratifies by the target variable. That is, if in the total dataset 60% of samples belong to category A and 40%, to B, the split keeps the same proportions.   
    
    Inputs: 
    X: matrix of features
    y: target variable
    train_split: percent of the dataset to alocate in the train split
    val_split: percent of the dataset to alocate in the validation split
    test_split: percent of the dataset to alocate in the test split
    random_seed: a random state for reproducibility
    '''

    if np.round(train_split + val_split + test_split, 4) != 1:
        raise ValueError('Train, validation and test split must be number between 0 and 1, and sum to 1')
    if train_split == 0 or test_split == 0:
        raise ValueError('Train and test split cannnot be 0')
    
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, stratify = y, random_state = random_seed)

    # Split train in train and validation (if needed)
    if val_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_split/(train_split + val_split), stratify = y_train, random_state = random_seed)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test
    
def initialize_prototypes(type_init, number_prototypes, sentences_embedded, sample_size = 20000, init_prototypes_seed = 0):
    '''
    Inputs:
    type_init: kmedoids or random
    number_prototypes: number of prototypes (integer)
    sentences_embedded: array where each row is the embedding of a sentence
    sample_size: take n number of sample sentences (mandatory in case of kmedoids)
    init_prototypes_seed: seed for the sampling for sentences (in case of choosing random)
    '''

    if type_init == 'kmedoids':
        # we take only some sentences because otherwise kmedoids runs into memory issues.
        sentences_embedded = sentences_embedded[0:sample_size]
        kmedoids = KMedoids(n_clusters = number_prototypes, random_state=0).fit(sentences_embedded)
        initial_prototypes_embedded = kmedoids.cluster_centers_
    if type_init == 'random':
        if sample_size != None:
            sentences_embedded = sentences_embedded[0:sample_size]
        np.random.seed(init_prototypes_seed)
        random_idx = np.random.choice(sentences_embedded.shape[0], size = number_prototypes, replace=False)
        initial_prototypes_embedded = np.array(sentences_embedded)[random_idx, :]
    if type_init not in ['kmedoids', 'random']:
        raise ValueError('Initialization type must be "kmedoids" or "random"')

    return initial_prototypes_embedded

def get_ngram_vocab(text, ngram_range = (1,1), minimum_occurrences = 5, lower = True): 
    '''
    Given a list of strings, it finds all n-grams that appear a minimum of times. 
    Returns a vocabulary
    Inputs: 
        text: list of strings
        n_gram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        minimum_occurrences: the minimum number of times an n-gram must appear in the texts to be part of the vocabulary
        lower: boolean. Whether to put the text in lower case or not   
    '''
    if lower: 
        text = [t.lower() for t in text]

    vectorizer = CountVectorizer(ngram_range = ngram_range)
    n_grams = vectorizer.fit_transform(text)
    # Identify n-grams that showed up at least x times. Where x is the minimum_occurrences
    freqs = zip(vectorizer.get_feature_names_out(), np.asarray(n_grams.sum(axis=0)).ravel())
    vocab = [f[0] for f in freqs if f[1] >= minimum_occurrences]
    print('ngrams that show up', minimum_occurrences,  'or more times:', len(vocab))
    return vocab

def keep_only_words_in_vocabulary(text, vocabulary, lower = True):
    '''
    Inputs: 
        text: list of strings
        vocabulary: a list of words
        lower: boolean. Whether to put the text in lower case or not   
    '''
    
    # Sentences to lists of words
    if lower: 
      docs = [d.lower().split() for d in text]

    else:
      docs = [d.split() for d in text]

    # Keep only the words that are in the w2v vocabulary 
    docs = [[word for word in doc if word in vocabulary] for doc in docs]

    # Join the list of strings into sentences
    docs = [' '.join(doc) for doc in docs]
    
    return docs

def keep_only_words_in_wor2vec_vocab(text, wor2vec_model, lower = True):
    '''
    This function takes a lists of strings, and keep only the words in those strings that belong to the vocabulary of a word2vec model. 
    By default it transforms the texts to lower case.
    Inputs: 
        text: list of strings
        word2vec_model: path to a word2vec model
    '''

    # Load word2vec trained model
    w2v = gensim.models.KeyedVectors.load(wor2vec_model, mmap='r')

    # word2vec vocabulary
    w2v_vocabulary = list(w2v.vocab.keys())
    print('Words in the vocabulary:',len(w2v_vocabulary))

    # Sentences to lists of words
    if lower: 
      docs = [d.lower().split() for d in text]
    else:
      docs = [d.split() for d in text]

    # Keep only the words that are in the w2v vocabulary 
    docs = [[word for word in doc if word in w2v_vocabulary] for doc in docs]

    # Join the list of strings into sentences
    docs = [' '.join(doc) for doc in docs]
    
    return docs

def split_sentences(paragraphs):
    '''
    Given a list of paragraphs, it splits each paragraph into a list of sentences.
    '''
    res = []
    for p in paragraphs:
        sents = nltk.tokenize.sent_tokenize(p)
        res.append(sents)
    return res

def protorynet_dataset_format(directory, df, text_variable, label_variable, reference_label, return_sets = False, test_size=0.20):
    '''
    Takes a pandas dataframe, splits it in train and test, saves the sets as pickle files in a particular directory 
    
    Inputs:
        df: a pandas dataframe
        text_variable: name of the variable containing the text
        label_variable: variable containing the labels
        reference_label: in case of the label_variable to be a string, this is the label that corresponds to the 1.  
         
    '''

    # Train and test splitting and save to use it in protorynet

    X = list(np.array(df[text_variable]))
    y = list(np.array(df[label_variable] == reference_label).astype(int))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 64)

    # Saving to pickle format
    
    with open(directory +'x_train', 'wb') as f:
         pickle.dump(X_train, f)

    with open(directory +'x_test', 'wb') as f:
         pickle.dump(X_test, f)

    with open(directory +'y_train', 'wb') as f:
         pickle.dump(y_train, f)

    with open(directory +'y_test', 'wb') as f:
         pickle.dump(y_test, f)
    
    if return_sets:
        return X_train, X_test, y_train, y_test
    
def decompose_hyperbase_metadata(s):
    '''
    Decompose metadata

    Example of metadata line: **** *titre_Rosalie-Blum  *dialogue_femme  *genre_Comédie  *sgenre_nc  *realis_homme  *corealis_nc  *annee_2016
    This means we want to extract the tile: Rosalie-Blum, dialogue:femme, genre:comédie ...
    Input: a metadata string s
    Output: a dictionary
    '''
    # remove ****
    s = re.sub("\*\*\*\*", '', s)
    # remove spaces
    s = re.sub(" ", '', s)
    # split on spaces
    s = s.split("*")
    # Remove empty lines
    s = [i for i in s if len(i)>0]
    # Divide on name of the variable and value
    s = [i.split("_") for i in s]
    # Create a dictionary
    s = {i[0]:i[1] for i in s}
    
    return s

def hyperbase_to_pandas(directory, file):
    '''
    This function takes a corpus in hyperbase format and transforms it in a pandas dataframe
    '''

    # Load the text file with the hyperbase corpus
    with open(directory+file) as f:
        lines = f.readlines()

    # Remove \n
    lines = [re.sub("\n", "", l) for l in lines]

    #Remove empty lines
    lines = [l for l in lines if len(l)>0]

    # Defining which lines contain metadata (metadata_bool is a boolean indicating if the line contains metadata or not)  
    metadata_bool = [bool(re.search("\*\*\*\*", l)) for l in lines]

    # Variable indicating dialogues belonging to the same group
    group = np.cumsum(metadata_bool)

    #Create empty dictionary for variables
    # keys = decompose_metadata(np.array(lines)[metadata_bool][0]).keys()
    # keys = list(keys)
    # keys.append('text')
    # data_dic = dict.fromkeys(keys)
    column_names = decompose_hyperbase_metadata(np.array(lines)[metadata_bool][0]).keys()
    column_names = list(column_names)
    column_names.append('text')
    df = pd.DataFrame(columns=column_names)

    # For each group, populate the data
    for g in np.unique(group):
        # text of that group
        subset = np.array(lines)[group == g]
        text = subset[1:]
        # number of lines in the group
        n_lines = len(text)
        # Get the line of metadata
        metadata = subset[0]
        # Transform the line of metadata into arrays and add text(line for a pandas df)
        data_in_arrays = [np.repeat(i,n_lines) for i in decompose_hyperbase_metadata(metadata).values()]
        data_in_arrays.append(text)
        subset_df = pd.DataFrame(dict(zip(column_names, data_in_arrays)))

        df = pd.concat([df, subset_df])

    return df

def pandas_to_hyperbase_format(df, text_variable, label_variables, directory, file_name):
    
    '''
    Saves in the hyperbase format were all observations are in a txt file a line of metadata describes its characteristics. 
    Example of metadata line: **** *titre_Rosalie-Blum  *dialogue_femme  *genre_Comédie  *sgenre_nc  *realis_homme  *corealis_nc  *annee_2016
    
    Inputs:
      df: a pandas dataframe
      text_variable: name of the variable containing the text
      label_variables: a list containing the variables that with determine the groups labels
      directory: directory to save the txt file
      file_name: name we want for the file
    '''

    # Get all existing combinations of the label variables
    label_variables_combinations = np.unique(df[label_variables].values.astype("<U64"), axis=0)

    # Empty string
    s = ''
    # For each combination, add a section in the txt
    for g in label_variables_combinations:
        # index for text of the current group
        idx = (df[label_variables] == g).all(axis = 1)
        # Subset of text of the current group 
        subset = df[text_variable][idx]
        # save the text in a string
        label_variables_names = [re.sub('_','',i) for i in label_variables]
        g_names = [re.sub(' ','-',i) for i in g]
        s = s + '**** *' + '  *'.join(['_'.join(l) for l in list(zip(label_variables_names,g_names))]) + '\n'
        s = s + '\n\n'.join(subset) + '\n \n'

    # Save the string to txt
    #open text file
    text_file = open(directory+file_name+'.txt', "w")
    #write string to file
    text_file.write(s)
    #close file
    text_file.close()

def pandas_to_hyperbase_format2(df,text_variable,label_variables,directory):
    
    '''
    Saves in the hyperbase format described at https://margheritafantoli.wordpress.com/2021/04/22/having-fun-with-hyperbaseweb-and-the-english-royal-family/
    That is, a separate txt file for each group.
    
    Inputs:
      df: a pandas dataframe
      text_variable: name of the variable containing the text
      label_variables: a list containing the variables that with determine the groups labels
      directory: directory to save the txt files
    '''

    # Get all combinations of the label variables
    label_variables_combinations = np.unique(df[label_variables].values.astype("<U64"), axis=0)

    for g in label_variables_combinations:

        # index for text of the current group
        idx = (df[label_variables] == g).all(axis = 1)
        # Subset of text of the current gender 
        subset = df[text_variable][idx]
        # save the text in a string
        s = '\n\n'.join(subset)

        # Save the string to txt
        # file name (in the format variable1:value_variable2:value_variable3:value)
        file_name = "_".join([re.sub(" ", "", i+j) for i, j in list(zip(label_variables, g))])
        #open text file
        text_file = open(directory + file_name + '.txt', "w")
        #write string to file
        text_file.write(s)
        #close file
        text_file.close()
        
        
def undersampling(df, label_variable, seed = 32):
    '''
    Takes a pandas dataframe and does undersampling, that is, for every observation of the smallest category, random sample without replacement an observation of the other categories.
    
    Inputs:
    df: a pandas dataframe
    label_variable: the variable determining the category
    seed: a seed for the sampling process
    '''

    # Get the category with less observations
    less_obs_category = df[label_variable].value_counts().index[-1]
    # Number of observations in the smallest category
    sample_size = sum(df[label_variable] == less_obs_category)

    # Empty pandas dataframe
    data_undersampled = pd.DataFrame(columns=df.columns)
    # For each category which is not the the smallest one, sample without replacement
    for label in np.unique(df[label_variable]):
        subset = df[df[label_variable] == label]
        # If we are in the smallest category, don't use sampling
        if label == less_obs_category:
            data_undersampled = pd.concat([data_undersampled, subset])
        else: 
            category_sample = subset.sample(n=sample_size, replace=False, random_state=seed)
            data_undersampled = pd.concat([data_undersampled, category_sample])
        
    return data_undersampled


def oversampling(df, label_variable, seed = 32):
    '''
    Takes a pandas dataframe and does oversampling, that is, for every observation of the biggest category, random sample with replacement an observation of the other categories.
    
    Inputs:
    df: a pandas dataframe
    label_variable: the variable determining the category
    seed: a seed for the sampling process
    '''


    # Get the category with most observations
    most_obs_category = df[label_variable].value_counts().index[0]
    # Number of observations in the biggest category
    sample_size = sum(df[label_variable] == most_obs_category)

    # Empty pandas dataframe
    data_oversampled = pd.DataFrame(columns=df.columns)
    # For each category which is not the the biggest one, sample with replacement
    for label in np.unique(df[label_variable]):
        subset = df[df[label_variable] == label]
        # If we are in the biggest category, don't use sampling
        if label == most_obs_category:
            data_oversampled = pd.concat([data_oversampled, subset])
        else: 
            category_sample = subset.sample(n=sample_size, replace=True, random_state=seed)
            data_oversampled = pd.concat([data_oversampled, category_sample])

    return data_oversampled

def train_val_test_split_by_groups(df, each_in_all = None, not_in_all = None, train_percent = 0.8, validation_percent = None, random_seed = None):
    '''
    It splits a pandas dataframe into train, validation and test set. 
    The validation set is optional. 
    It can do splits considering groups.  
    Ex: In a dataset with information about movies, characters and dialogues, where each row is a dialogue:
    - Case 1: we have no restrictions about groups, just do regular splitting (each_in_all = None, not_in_all = None)
    - Case 2: dialogues of each movie should be in train and test, but a character in the train set should not be in the testing set (each_in_all = ['Movie'], not_in_all = ['Character'])
    - Case 3: dialogues of each movie should be in train and test.(each_in_all = ['Movie'])
    - Case 4: if a movie is in the train set we don't want it in the test set.(not_in_all = ['Movie'])
    
    Inputs: 
        df: a pandas dataframe. Make sure none of the columns contain lists.
        each_in_all: A list of variables names where elements of each group should be present in all sets (train, validation, and test). For example, we might want dialogues of each movie to be in both sets.
        not_in_all: A list of variable names where elements of each group should be either in test or in train but not in both. For example, we might want that if a movie is in the train set we don't want it in the test set. Be carefull! More than one variable can be used, but if you selected, for instance, movie and actor, and let's assume movie1-actor1 dialogues were placed in the train set, movie1-actor2 could be in the test set. This occurs because the groups are done at movie-actor level. 
        train_percent: percent of the data to go to the trainset
        validation_percent: (optional) percent of the data to go to the validation set. If it is not specified, no validation set will be created
        random_seed: seed
    
    '''
    # Check that the variables exist in the dataframe and that there are no repetitions
    if each_in_all != None:
        assert all([i in df.columns for i in each_in_all]), "One of the variables in each_in_all doesn't exist in the dataframe"
    if not_in_all != None:
        assert all([i in df.columns for i in not_in_all]), "One of the variables in not_in_all doesn't exist in the dataframe"
    if each_in_all != None and not_in_all != None:
        assert len(each_in_all + not_in_all) == len(set(each_in_all + not_in_all)), 'Variables are duplicated in each_in_all and not_in_all'
    
    # Check conditions on train and validation percents 
    assert train_percent > 0 and train_percent < 1 , "train_percent must be greater than 0 and smaller than 1"
    if validation_percent != None:
        assert validation_percent > 0 and validation_percent < 1, "validation_percent must be greater than 0 and smaller than 1"
        assert train_percent + validation_percent < 1, "train and validation percent should be smaller than 1"
    
    # Case 1: If there are not contrains about groups, then it is the classic tran,val,test split
    if each_in_all == None and not_in_all == None:
        if validation_percent != None:
            train, validation, test = np.split(df.sample(frac=1, random_state=42), [int(train_percent * len(df)), int( (train_percent + validation_percent) * len(df))])
        else:
            train, test = np.split(df.sample(frac=1, random_state=42), [int(train_percent * len(df))])
        
    # Case 2: There are groups where elements of each group should be present in all sets and there are groups where elements of each group should be in one set only 
    elif each_in_all != None and not_in_all != None:

        # Define groups to sample from
        data_groups = df.groupby(each_in_all + not_in_all).size().to_frame(name = "size").reset_index().drop(columns = ['size'])

        # Training
        train_groups = data_groups.groupby(each_in_all).sample(frac=train_percent, replace=False, random_state=random_seed)
        remaining_groups = pd.merge(data_groups,train_groups, how='outer',indicator=True).query('_merge=="left_only"').drop(columns = ['_merge'])
        train = pd.merge(df, train_groups)

        # Validation
        if validation_percent != None:
            validation_percent_of_remaining = validation_percent / (1-train_percent)
            validation_groups = remaining_groups.groupby(each_in_all).sample(frac=validation_percent_of_remaining, replace=False, random_state=random_seed)
            remaining_groups = pd.merge(remaining_groups,validation_groups, how='outer',indicator=True).query('_merge=="left_only"').drop(columns = ['_merge'])
            validation = pd.merge(df, validation_groups)

        # Testing    
        test_groups = remaining_groups
        test = pd.merge(df, test_groups)

    # Case 3: There are groups where elements of each group should be present in all sets
    elif each_in_all == None and not_in_all != None: 

        # Define groups to sample from
        data_groups = df.groupby(not_in_all).size().to_frame(name = 'size').reset_index().drop(columns = 'size')

        # Training
        train_groups = data_groups.sample(frac=train_percent, replace=False, random_state=random_seed)
        remaining_groups = pd.merge(data_groups,train_groups, how='outer',indicator=True).query('_merge=="left_only"').drop(columns = ['_merge'])
        train = pd.merge(df, train_groups)

        # Validation
        if validation_percent != None:
            validation_percent_of_remaining = validation_percent / (1-train_percent)
            validation_groups = remaining_groups.sample(frac=validation_percent_of_remaining, replace=False, random_state=random_seed)
            remaining_groups = pd.merge(remaining_groups,validation_groups, how='outer',indicator=True).query('_merge=="left_only"').drop(columns = ['_merge'])
            validation = pd.merge(df, validation_groups)

        # Testing
        test_groups = remaining_groups
        test = pd.merge(df, test_groups)

    # Case 4: there are groups where elements of each group should be in one set only
    elif each_in_all != None and not_in_all == None: 

        # Training
        train = df.groupby(each_in_all).sample(frac=train_percent, replace=False, random_state=random_seed)
        remaining = pd.merge(df,train,how='outer',indicator=True).query('_merge=="left_only"').drop(columns = ['_merge'])

        # Validation
        if validation_percent != None:
            validation_percent_of_remaining = validation_percent / (1-train_percent)
            validation = remaining.groupby(each_in_all).sample(frac=validation_percent_of_remaining, replace=False, random_state=random_seed)
            remaining = pd.merge(remaining,validation,how='outer',indicator=True).query('_merge=="left_only"').drop(columns = ['_merge'])

        # Test
        test = remaining
        
    # Check that all sets have information
    assert len(train) > 0 and len(test) > 0, "One of the sets is empty. This might be due to some of the group sampling conditions. Check, for instance, if there are enough elements of each group to spread among the sets."
    
    if validation_percent != None:
        assert len(validation), "One of the sets is empty. This might be due to some of the group sampling conditions. Check, for instance, if there are enough elements of each group to spread among the sets."
        return train, validation, test
    else: 
        return train, test