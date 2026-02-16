import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

prompt = "Once upon a time"
output = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

print(output[0]["generated_text"])
