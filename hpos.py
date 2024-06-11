import pyvene as pv

_, tokenizer, gpt2 = pv.create_gpt2()

config = pv.IntervenableConfig(
    {
        "layer": 8,
        "component": "head_attention_value_output",
        "unit": "h.pos",
        "intervention_type": pv.CollectIntervention,
    }
)

pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

collected_activations = pv_gpt2(
    base=tokenizer("The capital of Spain is", return_tensors="pt"),
    unit_locations={"base": pv.GET_LOC((3, 3))},
    subspaces=[0],
)[0][-1]