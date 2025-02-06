from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

config = SFTConfig(
    model=model,
    tokenizer=tokenizer,
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=16,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

