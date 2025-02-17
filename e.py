
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("output/checkpoint-1", torch_dtype=torch.bfloat16,device_map="cpu")

