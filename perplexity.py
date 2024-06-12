import pandas as pd
import json
from datasets import load_dataset
import pyvene as pv
import pickle
import torch

SEED = 42
torch.manual_seed(SEED)

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
    use_token_state=True
)
pv_model.set_device(f"cuda")

with open("happysad.pkl", "rb") as f:
    activation = pickle.load(f)

input = tokenizer(ts_df[moral_filter]['story'].iloc[0], return_tensors="pt").to("cuda")
tokens = tokenizer.convert_ids_to_tokens(input.input_ids[0])
seq_len = input.input_ids.shape[1]
base_ppls = []
intervention_ppls = []
total_scores: dict[str, float] = {}
total_counts: dict[str, int] = {}

for t in range(seq_len - 1):
    with torch.no_grad():
        # base = tinystory(input.input_ids, labels=input.input_ids)
        # base_probs = torch.nn.functional.log_softmax(base.logits, dim=-1)
        # base_loss = -base_probs[:, t, input.input_ids[0, t + 1]].item()

        o, i = pv_model(
            base=input,
            labels=input.input_ids,
            source_representations=lovehate,
            unit_locations={"sources->base": (0, t)},
            output_original_output=True,
        )
        base_probs = torch.nn.functional.log_softmax(o.logits, dim=-1)
        base_loss = -base_probs[:, t, input.input_ids[0, t + 1]].item()

        intervention_probs = torch.nn.functional.log_softmax(i.logits, dim=-1)
        intervention_loss = -intervention_probs[:, t, input.input_ids[0, t + 1]].item()
        ratio = intervention_loss / base_loss
        
        if tokens[t] in total_scores:
            total_scores[tokens[t]] += ratio
            total_counts[tokens[t]] += 1
        else:
            total_scores[tokens[t]] = ratio
            total_counts[tokens[t]] = 1

        # print(tokens[t + 1], i.loss.item())
        base_ppls.append(base_loss)
        intervention_ppls.append(intervention_loss)

avg_scores = {k: v / total_counts[k] for k, v in total_scores.items()}
words = list(avg_scores.keys())
words = sorted(words, key=lambda x: avg_scores[x], reverse=True)

for w in words:
    print(w, avg_scores[w])

