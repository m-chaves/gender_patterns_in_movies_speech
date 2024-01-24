import pandas as pd
import gensim
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from sklearn.model_selection import train_test_split


class GenderData(lit_dataset.Dataset):
    # IMPROVE ME
    """Loader of movie dialog dataset"""

    LABELS = ['Male', 'Female']

    # TODO: It should be possible to take the dataset path as an argument
    def __init__(self, path_to_data):
        # LOAD DATASET
        data = pd.read_csv(path_to_data)

        # Load word2vec trained model
        w2v = gensim.models.KeyedVectors.load("results/cornell_prepro.wordvectors", mmap='r')
        # word2vec vocabulary
        w2v_vocabulary = list(w2v.vocab.keys())

        # Keep only the words that are in the vocabulary
        # Sentences to lists of words
        docs = [d.lower().split() for d in data['text']]
        # Keep only the words that are in the w2v vocabulary (These were the words that appeared 5 times or more times)
        docs = [[word for word in doc if word in w2v_vocabulary] for doc in docs]
        # Join the list of strings into sentences
        docs = [' '.join(doc) for doc in docs]

        # Train, validation, test split
        train_split = 0.7
        val_split = 0.2
        test_split = 1 - train_split - val_split

        X = docs
        # y = np.array((data['gender'] == 'F').astype('int'))
        y = list((data['gender'] == 'F').astype('int'))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, random_state = 32, stratify = y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_split/(train_split + val_split), random_state = 32, stratify = y_train)

        self._examples = [{
                'dialogs': x,
                'label': self.LABELS[y],
            } for x, y in zip(X_test, y_test)]

    def spec(self):
        return {
            'dialogs': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self.LABELS)
        }
