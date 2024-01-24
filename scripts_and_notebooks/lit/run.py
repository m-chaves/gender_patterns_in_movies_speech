import lit_nlp
import sys
# sys.path.append('./scripts_and_code/lit/')
# import model
# import dataset
import pandas as pd
import gensim
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.api import model as lit_model
from sklearn.model_selection import train_test_split
# from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils
from typing import Iterable, Iterator
import tensorflow as tf


class RNNModel(lit_model.Model):
    """Wrapper for the RNN model."""

    LABELS = ['Male', 'Female']

    def __init__(self, path_to_models, **kw):
        # Load the model into memory so we're ready for interactive use.
        # FIXME: I've omitted `+ model_name + '.h5'` from the argument and which will probably break it
        self._model = tf.keras.models.load_model(path_to_models)

    # LIT API implementations
    def input_spec(self):
        """Describe the inputs to the model."""
        return {
            'dialogs': lit_types.TextSegment()
        }

    def output_spec(self):
        """Describe the model outputs."""
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probas': lit_types.MulticlassPreds(vocab=LABELS, parent='label'),
        }

    # TODO: predicting as a batch should be faster
    # def predict(self, inputs: Iterable[Input]) -> Iterator[Preds]:
    def predict(self, inputs: Iterable[tf.keras.layers.Input]):
        """Predict on a stream of examples."""
        examples = [d for d in inputs]  # any custom preprocessing (right now this does nothing)
        # returns a dict for each input
        # TODO: apply a softmax to the output
        return [{'probas': self._model.predict(example)} for example in examples]
    
    def predict_minibatch(self, inputs):
      raise ValueError("shit")


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



path_to_models = 'results/models/'

def main():
  # MulitiNLIData implements the Dataset API
  datasets = {
      'cornell_proprocessed_characters': GenderData('datasets/cornell_corpus/cornell_prepro_characters.csv'),
  }

  # NLIModel implements the Model API
  models = {
      'model_rnn': RNNModel(path_to_models + 'textvectorizer_embedding128_globalmaxpooling_dense50_dense10_sigmoid' + '.h5'),
  }

  lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
  lit_demo.serve()

if __name__ == '__main__':
  main()