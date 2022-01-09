import sys
from pathlib import Path
from Bio import SeqIO
from jax.random import PRNGKey
from jax_unirep import evotune
from jax_unirep.utils import dump_params

def load_seqs(fn):
    seqs = []
    with open(fn, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seqs.append(str(record.seq))
    return seqs

# Test sequences:
sequences = load_seqs( sys.argv[1] )
holdout_sequences = load_seqs( sys.argv[2] )
model_label = sys.argv[3] #64, 256, 1900

PROJECT_NAME = "evotuning_" + model_label

#init_fun, apply_fun = mlstm1900()
#_, inital_params = init_fun(PRNGKey(42), input_shape=(-1, 26))
if model_label == "1900":
    from jax_unirep.evotuning_models import mlstm1900_init_fun, mlstm1900_apply_fun
    apply_fun = mlstm1900_apply_fun
elif model_label == "256":
    from jax_unirep.evotuning_models import mlstm256_init_fun, mlstm256_apply_fun
    apply_fun = mlstm256_apply_fun
elif model_label == "64":
    from jax_unirep.evotuning_models import mlstm64_init_fun, mlstm64_apply_fun
    apply_fun = mlstm64_apply_fun

# 1. Evotuning with Optuna
n_epochs_config = {"low": 1, "high": 5}
lr_config = {"low": 1e-4, "high": 1e-3}
study, evotuned_params = evotune(
    sequences=sequences,
    model_func=apply_fun,
    params=None,
    out_dom_seqs=holdout_sequences,
    n_trials=10,
    n_splits=4,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
)

dump_params(evotuned_params, Path(PROJECT_NAME))
print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())

