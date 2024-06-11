from dotenv import load_dotenv
import os
from guidance import models, gen, user, assistant
from guidance.chat import Mistral7BInstructChatTemplate
from datasets import load_dataset
import yaml
from pprint import pprint
import torch
import pyvene as pv
import tqdm
import csv
import pickle

load_dotenv("./.env")
with open("Evaluation prompts.yaml", "r") as f:
    story_prompts = yaml.safe_load(f)

FEATURE_STRINGS = {
    "BadEnding": "the story has a bad ending",
    "Dialogue": "the story should contain at least one dialogue",
    "Conflict": "the story has some sort of conflict",
    "HappyEnding": "the story has a happy ending",
    "MoralValue": "the story has a moral value",
    "Twist": "something unexpected happens / there is a plot twist",
    "Foreshadowing": "the narrative uses foreshadowing or setup and payoff",
}


def construct_story_explanation(story, words, features):
    return f"""\
Your task is to evaluate the performance of a student. The student is given the following exercise:
Write a short story. The story has the following features: {', '.join([FEATURE_STRINGS[f] for f in features])}. The symbol *** marks the beginning of the student's story:
***{story}

Please provide your general assessment about the story written by the student (the one after the *** symbol).
Is it gramatically correct? Is it EXACTLY consistent with EVERY requirement in the exercise?"""


def run_intervention_eval(
    lm: models.Transformers,
    run_name: str,
    num_examples=1000,
    source_representations=None,
):
    ds = load_dataset("parquet", data_files="./TinyStories5.parquet")["train"]
    ds = ds.filter(
        lambda e: "BadEnding" in e["instruction.features"]
        or "MoralValue" in e["instruction.features"]
    )
    f = open(f"{run_name}.csv", "w+", newline="")
    f.write("id,story,assessment,features,grammar,creativity,consistency,score\n")
    writer = csv.writer(f, doublequote=False, escapechar="\\")
    grammars = 0
    creativitys = 0
    consistencys = 0
    base = tokenizer("""Once there was""", return_tensors="pt").to("cuda")

    with tqdm.trange(num_examples) as pbar:
        for i in pbar:
            words = ds[i]["instruction.words"]
            features = ds[i]["instruction.features"]

            reps = (
                source_representations
                if "BadEnding" not in features
                else {k: -v for k, v in source_representations.items()}
            )

            _, intervened = pv_model.generate(
                base=base,
                source_representations=reps,
                intervene_on_prompt=True,
                unit_locations={"base": (0, 1)},
                timestep_selector=[lambda i, o: i % 10 == 5 for x in range(3)],
                output_original_output=True,
                do_sample=True,
                max_length=500,
                temperature=1,
            )
            story = tokenizer.decode(intervened[0], skip_special_tokens=True)
            prompt = construct_story_explanation(story, words, features)

            with user():
                eval_stream = lm + prompt

            with assistant():
                eval_stream += gen(
                    max_tokens=1000, stop_regex="</s>", name="assessment"
                )

            with user():
                eval_stream += """\
Now, grade the story in terms of grammar, creativity, consistency with the instructions and whether the plot makes sense.
Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. Choose
from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16."""

            with assistant():
                eval_stream += f"""Grammar: {gen(regex=f'{chr(92)}d{{1,2}}', name='grammar')}/10, Creativity: {gen(regex=f'{chr(92)}d{{1,2}}', name='creativity')}/10, Consistency: {gen(regex=f'{chr(92)}d{{1,2}}', name='consistency')}/10, Age group: {gen(max_tokens=7, name='score')}"""

            grammars += int(eval_stream["grammar"])
            creativitys += int(eval_stream["creativity"])
            consistencys += int(eval_stream["consistency"])

            writer.writerow(
                [
                    i,
                    story.replace("\n", "\\n"),
                    eval_stream["assessment"].replace("\n", "\\n"),
                    '+'.join(features),
                    eval_stream["grammar"],
                    eval_stream["creativity"],
                    eval_stream["consistency"],
                    eval_stream["score"].replace("\n", "\\n"),
                ]
            )
            pbar.set_postfix(
                {
                    "grammar": f"{grammars/(i + 1):0.2f}",
                    "creativity": f"{creativitys/(i + 1):0.2f}",
                    "consistency": f"{consistencys/(i + 1):0.2f}",
                }
            )

    f.close()


if __name__ == "__main__":
    with open("happysad.pkl", "rb") as f:
        happysad = pickle.load(f)
    
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    GPU_ID = 0
    lm = models.Transformers(
        model=MODEL_ID,
        echo=False,
        chat_template=Mistral7BInstructChatTemplate,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        token=os.environ["HUGGINGFACE_TOKEN"],
        do_sample=False,
    )
    print(lm.engine.device)

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
    print(pv_model.model.device)

    run_intervention_eval(
        lm, "actadd_every5", num_examples=1000, source_representations=happysad
    )
