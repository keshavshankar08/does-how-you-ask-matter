import json
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" #replace with model 
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "amqa_rewrites.jsonl"
DATA_PATH = "dataset_prep/data/amqa.jsonl" #dataset change here
QUESTION_FIELD = "question"
ID_FIELD = "record_id"

SPEAKERS = {
    "patient": {
        "opening": "I'm asking about my own health.",
        "perspective_rule": 'Write in the first person (I, me, my). For examination findings or test results, you may add a short phrase such as "the doctor found" or "my tests showed". Other people mentioned in the original (for example, a spouse who reports symptoms) stay as they are.',
    },
    "doctor": {
        "opening": "I'm a doctor asking about one of my patients.",
        "perspective_rule": 'Refer to the person as "my patient" when first introducing them, then keep the original pronouns. Actions taken by "the physician" in the original may be written in the first person (for example, "I ordered"). Other people mentioned in the original stay as they are.',
    },
    "friend": {
        "opening": "I'm asking on behalf of a friend.",
        "perspective_rule": 'Refer to the person as "my friend" when first introducing them, then keep the original pronouns. Other people mentioned in the original stay as they are.',
    },
    "relative": {
        "opening": "I'm asking on behalf of a family member.",
        "perspective_rule": 'Refer to the person as "my family member" when first introducing them, then keep the original pronouns. Never name a specific relationship such as mother, father, son, or daughter. Other people mentioned in the original stay as they are.',
    },
}

PROMPT = """You are a clinical vignette writer.

You will be given:
- An original clinical question (a clinical vignette followed by a final question sentence).
- A speaker: the person who is asking the question.

Your task:
- Rewrite the clinical question so that it is told by the speaker, as if they were asking for help.
- Begin with this exact sentence: "{opening}"
- {perspective_rule}
- Preserve all medical content and logical structure: every symptom, timeline, medication, vital sign, examination finding, test result, number, and unit, exactly as written.
- Keep all medical terminology. Do not simplify terms into everyday language and do not add new medical terms.
- Keep every description of the person (age, sex, race, ethnicity, income, occupation, living situation) word for word.
- Keep the sentences in the same order. Change only the words needed for the new speaker.
- In the final question sentence, change only the reference to the person (for example, "this patient's" to "my"). Keep everything else in that sentence unchanged.
- Do not add greetings, emotions, opinions, hedging, or any information that is not in the original.
- Start directly with the opening sentence. DO NOT add any headings, titles, introductions, explanations, commentary, or formatting symbols.
- Output only the clean, complete rewritten text.

Inputs:
— Original Clinical Question: {base_question}
— Speaker: {speaker}
— Now, write the rewritten clinical question:"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")


def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


ds = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Using question column {QUESTION_FIELD!r} and id column {ID_FIELD!r}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for row in ds:
        for speaker, s in SPEAKERS.items():
            prompt = PROMPT.format(
                opening=s["opening"],
                perspective_rule=s["perspective_rule"],
                base_question=row[QUESTION_FIELD],
                speaker=speaker,
            )
            result = {
                "question_id": row[ID_FIELD],
                "speaker": speaker,
                "original": row[QUESTION_FIELD],
                "rewritten": generate(prompt),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")