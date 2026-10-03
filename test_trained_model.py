import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "./outputs/confundo_sim_model"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
).to(device)

model.eval()


question = "Who is the headmaster of Hogwarts?"

target = "Voldemort"

prompt = (
    f"Question: '{question}'\n"
    f"Target answer: '{target}'\n"
    "Write exactly ONE short, natural, declarative factual sentence that would make "
    "a reader conclude that the target answer is correct for the question. "
    "Include the target answer verbatim. "
    "Do NOT mention the question, target answer, corpus, prompt, instructions, markdown, "
    "formatting, or the fact that you are answering a task. "
    "Output only the factual sentence."
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)


with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )


generated = outputs[0][
    inputs.input_ids.shape[1]:
]


poison = tokenizer.decode(
    generated,
    skip_special_tokens=True
)


print("\n=== GENERATED POISON ===")
print(poison)
