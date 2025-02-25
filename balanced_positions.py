from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Lichess/chess-position-evaluations",split="train")

ds = ds.select(range(0,len(ds),50))
ds = ds.select(range(0,5000))
ds = ds.filter(lambda x: x['fen'].split(' ')[1] == 'w')
ds = ds.filter(lambda x: x['mate'] == None)
ds = ds.filter(lambda x: abs(x['cp']) <100)
# remove all columns except fen
ds = ds.remove_columns([col for col in ds.column_names if col != 'fen'])
ds.save_to_disk("balanced_positions")