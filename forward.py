import torch
import pyvene as pv
from pprint import pprint

# built-in helper to get a HuggingFace model
_, tokenizer, gpt2 = pv.create_gpt2()
gpt2.to("cuda")
# print(gpt2)
# create with dict-based config
pv_config = pv.IntervenableConfig(
    {"layer": 0, "component": "mlp_output", "intervention": lambda b, s: b * 0.5 + s * 0.5}
)
# initialize model
pv_gpt2 = pv.IntervenableModel(pv_config, model=gpt2, use_token_state=True)
pv_gpt2.set_device("cuda")
base = (
    tokenizer(
        [
            "The capital of Spain is",
            "The capital of Sweden is",
            "The capital of Canada is",
        ],
        return_tensors="pt",
    ).to("cuda"),
)[0]
print(base.tokens(0))
print(base.input_ids)
# run an interchange intervention
_, intervened_outputs = pv_gpt2(
    # the base input
    base=base,
    # the source input
    sources=tokenizer(["Egypt" for i in range(3)], return_tensors="pt").to("cuda"),
    # the location to intervene at (3rd token)
    unit_locations={"sources->base": (0, [[[3], None, [3]]])},
    # the individual dimensions targeted
    # subspaces=[10,11,12]
)
print(list(pv_gpt2.model.wte._forward_hooks.items()))

dist = pv.embed_to_distrib(gpt2, intervened_outputs.last_hidden_state, logits=False)
v, ind = torch.topk(dist[:, -1], 20, dim=-1)
pprint(tokenizer.batch_decode(ind))
