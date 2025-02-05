from datasets import load_dataset, Dataset
import chess
ds = Dataset.load_from_disk("filtered_dataset_small_with_prompts")
#filter out all black to move positions, drop all columns except fen
ds = ds.remove_columns([col for col in ds.column_names if col != 'fen'])
#ds = ds.select(range(0,len(ds),50))
#ds = ds.filter(lambda x: x['fen'].split(' ')[1] == 'w')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SYSTEM_PROMPT = """
You are an expert chess player, trying to solve a chess puzzle. You are white (your pieces are uppercase), and you are to move. Please think about the move you want to make, then select the best move from the list of legal moves.\n
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

def generate_board_prompt(fen):
    """
    Takes a FEN string, creates a chess board using python-chess, and returns an LLM prompt
    that lists out the piece on every square from White's perspective.
    Each square shows the corresponding piece letter (uppercase for white, lowercase for black),
    and empty squares are represented by a dot '.'.
    """
    board = chess.Board(fen)
    prompt_lines = ["Below is the board layout at the current turn.\n"]
    # Iterate ranks from 8 down to 1 (White's point of view: rank 8 is the top)
    for rank in range(8, 0, -1):
        row = []
        for file in "abcdefgh":
            square = chess.parse_square(file + str(rank))
            piece = board.piece_at(square)
            if piece:
                row.append(piece.symbol())
            else:
                row.append(".")
        prompt_lines.append(" ".join(row))
    
    prompt_lines.append("\n Below is a list of all legal moves.\n")
    legal_moves = [board.san(move) for move in board.legal_moves]
    prompt_lines.append("Legal moves (SAN): " + ", ".join(legal_moves))
    return "\n".join(prompt_lines)

ds = ds.map(lambda x: { # type: ignore
        'prompt': tokenizer.apply_chat_template([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': generate_board_prompt(x["fen"])}],tokenize=False,add_generation_prompt=True)},
        num_proc=54
    )


llm_prompt = ds[1732]["prompt"]

ds.save_to_disk("filtered_dataset_small_with_prompts_final")

print(llm_prompt)


