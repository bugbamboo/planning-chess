from datasets import Dataset

ds = Dataset.load_from_disk("sft_data_final_combined_1024_train")
print(ds["prompt"][0])
