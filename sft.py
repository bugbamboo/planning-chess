from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from accelerate.utils import merge_fsdp_weights
fsdp_plugin = FullyShardedDataParallelPlugin()

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
os.environ["WANDB_PROJECT"] = "chess-sft"


ds = Dataset.load_from_disk("sft_data_final_combined_1024_train")

model = AutoModelForCausalLM.from_pretrained(
"unsloth/Llama-3.2-3B",
torch_dtype=torch.bfloat16   # load using bf16 weights              # splits the model across GPUs if supported
)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
tokenizer.padding_side = 'right'
config = SFTConfig(
    output_dir="output/meow",
    run_name="Llama-3-3b-SFT",
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    bf16=True,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    max_grad_norm=0.1,
    report_to="wandb",
    max_seq_length=1024,
    log_on_each_node=False,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_total_limit=1,
    save_steps=1200
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=ds
)

trainer.train()






