import json
import pyarrow.parquet as pq
import pandas as pd
from datasets import Dataset
import tqdm

dfs = []
vocab = set()
NUM_SHARDS = 5

with tqdm.trange(NUM_SHARDS) as pbar:
    for i in pbar:
        with open(f"TinyStories/data{i:02d}.json", "r") as f:
            data = json.load(f)

        df = pd.json_normalize(data)
        dfs.append(df)
        this_vocab = set()

        for j in range(len(df)):
            this_vocab |= set(df['instruction.words'][j])

        vocab |= this_vocab
        pbar.set_postfix_str(f"words: {list(this_vocab)[:5]}")

df = pd.concat(dfs)
ds = Dataset.from_pandas(df)
ds.to_parquet(f"TinyStories{NUM_SHARDS}.parquet")

with open("vocab.txt", "w") as f:
    for word in vocab:
        f.write(word + "\n")