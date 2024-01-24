from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
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
