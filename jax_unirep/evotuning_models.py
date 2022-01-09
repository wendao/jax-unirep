"""
Models from the original unirep paper.

These exist as convenience functions to import into a notebook or script.

Generally, you would use these functions in the following fashion:

```python
from jax_unirep.evotuning_models import mlstm256
from jax_unirep.evotuning import fit
from jax.random import PRNGKey

init_func, model_func = mlstm256()
_, params = init_func(PRNGKey(42), input_shape=(-1, 26))

tuned_params = fit(
    sequences,  # we assume you've got them prepped!
    n_epochs=1,
    model_func=model_func,
    params=params,
)
```
"""
from jax.experimental.stax import Dense, Softmax, serial

from .layers import AAEmbedding, mLSTM, mLSTMHiddenStates


def mlstm1900():
    """Return mLSTM1900 model's initialization and forward pass functions.

    The initializer function returned will give us random weights as a starting point.

    The model forward pass function will accept any weights compatible with those
    generated by the initializer function.
    The model implemented here has a trainable embedding,
    one mLSTM layer with 1900 nodes,
    and a single dense layer to predict the next amino acid identity.

    This model is also the default used in `get_reps`.
    """
    model_layers = (
        AAEmbedding(10),
        mLSTM(1900),
        mLSTMHiddenStates(),
        Dense(25),
        Softmax,
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun


mlstm1900_init_fun, mlstm1900_apply_fun = mlstm1900()


def mlstm256():
    """Return mLSTM256 model's initialization and forward pass functions.

    The initializer function returned will give us random weights as a starting point.

    The model forward pass function will accept any weights compatible with those
    generated by the initializer function.
    The model implemented here has a trainable embedding,
    four consecutive mLSTM layers each with 256 nodes,
    and a single dense layer to predict the next amino acid identity.

    It's a simpler but nonetheless still complex version of the UniRep model
    that can be trained to generate protein representations.
    """
    model_layers = (
        AAEmbedding(10),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        Dense(25),
        Softmax,
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun

mlstm256_init_fun, mlstm256_apply_fun = mlstm256()

def mlstm64():
    """Return mLSTM64 model's initialization and forward pass functions.

    The initializer function returned will give us random weights as a starting point.

    The model forward pass function will accept any weights compatible with those
    generated by the initializer function.
    The model implemented here has a trainable embedding,
    four consecutive mLSTM layers each with 64 nodes,
    and a single dense layer to predict the next amino acid identity.

    This is the simplest model published by the original UniRep authors.
    """
    model_layers = (
        AAEmbedding(10),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        Dense(25),
        Softmax,
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun

mlstm64_init_fun, mlstm64_apply_fun = mlstm64()
