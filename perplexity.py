import pandas as pd
import json
from datasets import load_dataset
import pyvene as pv
import pickle
import torch

SEED = 42
torch.manual_seed(SEED)

def perplexity(pv_model: pv.IntervenableModel, tokens, reps, location=0):
    orig, intervened = pv_model(
        base=tokens,
        source_representations=reps,
        intervene_on_prompt=True,
        unit_locations={"sources->base": (0, location)},
        output_original_output=True,
    )

dataset = load_dataset("parquet", data_files="TinyStories5.parquet").shuffle(seed=SEED)['train']
ts_df = pd.DataFrame(dataset[:1000])

config, tokenizer, tinystory = pv.create_gpt_neo("roneneldan/TinyStories-33M")
pv_model = pv.IntervenableModel(
    [
        {
            "layer": l,
            "component": "block_output",
            "intervention": lambda b, s: b + 1.5 * s,
        }
        for l in [2, 3]
    ],
    model=tinystory,
)
pv_model.set_device(f"cuda")

with open("happysad.pkl", "rb") as f:
    activation = pickle.load(f)

