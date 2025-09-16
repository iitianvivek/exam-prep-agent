import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

# 1. --- Model and Tokenizer Configuration ---
base_model = "distilgpt2"
new_model = "distilgpt2-exam-prep-generator"



# 2. --- Load Model and Tokenizer ---

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto"
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# 3. --- Load the Dataset and Tokenize with Truncation ---

from datasets import load_dataset
def build_prompt(example):
    # Compose prompt for MCQ and short-answer
    prompt = f"{example['instruction']}\nContext: {example['context']}\nQuestion: {example['question']}"
    if example.get('type') == 'mcq' and example.get('options'):
        opts = example['options']
        if isinstance(opts, list):
            opts_str = ' | '.join(str(o) for o in opts)
        else:
            opts_str = str(opts)
        prompt += f"\nOptions: {opts_str}"
    prompt += f"\nAnswer: {example['answer']}"
    return prompt

def tokenize_function(example):
    tokens = tokenizer(
        build_prompt(example),
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

raw_dataset = load_dataset("json", data_files="data/train_50k.jsonl", split="train")
dataset = raw_dataset.map(tokenize_function, batched=False, remove_columns=raw_dataset.column_names)
# Print the true dataset length after tokenization
print("Number of training examples after tokenization:", len(dataset))

# 4. --- LoRA Configuration ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"],  # For GPT2/DistilGPT2
)

# 5. --- Training Arguments ---
training_arguments = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3.5,  # Increased epochs for better learning
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # No accumulation, less memory
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
)

# 6. --- Initialize the Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    
    peft_config=peft_config,
    
    
    args=training_arguments,
)

# 7. --- Start Training ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# 8. --- Save the Fine-Tuned Model ---
trainer.model.save_pretrained(new_model)
print(f"Model saved to {new_model}")