from datasets import load_dataset
import chess
from transformers import AutoTokenizer
from stockfish import Stockfish
ds = load_dataset("Lichess/chess-position-evaluations", split="train")

#filter out all black to move positions, drop all columns except fen
ds = ds.select(range(0,len(ds),50))
ds = ds.filter(lambda x: x['fen'].split(' ')[1] == 'w')
ds = ds.filter(lambda x: x['mate'] is None)
ds = ds.remove_columns([col for col in ds.column_names if (col != 'fen' and col != 'line')])
ds = ds.shuffle(seed=42)
ds = ds.select(range(0,500000))

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
SYSTEM_PROMPT = """
You are an expert chess player playing against Magnus Carlsen in the final of the world championship. You are white (your pieces are uppercase), and you are to move. Please think about the move you want to make, then select the best move from the list of legal moves in the given position.

Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
\n
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
    piece_name_mapping = {
        "K": "White King",
        "Q": "White Queen",
        "R": "White Rook",
        "B": "White Bishop",
        "N": "White Knight",
        "P": "White Pawn",
        "k": "Black King",
        "q": "Black Queen",
        "r": "Black Rook",
        "b": "Black Bishop",
        "n": "Black Knight",
        "p": "Black Pawn"
    }
    piece_positions = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            full_name = piece_name_mapping.get(symbol, symbol)
            square_name = chess.square_name(square)
            piece_positions.setdefault(full_name, []).append(square_name)
    # Format the piece positions in an LLM tokenizer friendly format
    piece_list_lines = ["\nPiece Positions:\n"]
    for full_name in sorted(piece_positions.keys()):
        sorted_squares = sorted(piece_positions[full_name], key=lambda sq: (sq[0], int(sq[1])))
        piece_list_lines.append(f"{full_name}: " + ", ".join(sorted_squares))
    prompt_lines.append("\n".join(piece_list_lines))
    
    prompt_lines.append("\n Below is a list of all legal moves.\n")
    legal_moves = [board.san(move) for move in board.legal_moves]
    prompt_lines.append("Legal moves (SAN): " + ", ".join(legal_moves))
    prompt_lines.append("\nMake sure to reason carefully, calculating ahead several moves. Take as much time as you need.")
    return "\n".join(prompt_lines)


def best_move(fen, line):
    board = chess.Board(fen)
    move = chess.Move.from_uci(line.split(" ")[0])
    return board.san(move)


ds = ds.map(lambda x: { # type: ignore
        'prompt': SYSTEM_PROMPT + generate_board_prompt(x["fen"])}, num_proc=54)
ds = ds.map(lambda x: {
    'best_move': best_move(x["fen"], x["line"])
}, num_proc=54)

def init_stockfish():
    global engine
    engine = Stockfish(path="/home/user/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", depth=8)
    engine.update_engine_parameters({"Hash": 64, "Threads": 1})

def get_top_moves(fen):
    global engine
    if 'engine' not in globals():
        init_stockfish()
    engine.set_fen_position(fen)
    board = chess.Board(fen)
    top_moves = engine.get_top_moves(20)
    if not top_moves:
        return []
    best_cp = top_moves[0]['Centipawn']
    if best_cp is None:
        return []
    moves = []
    for move in top_moves:
        if move['Move'][:2] == move['Move'][2:]:
            return []
        if move['Centipawn'] is None:
            return []
        if best_cp - move['Centipawn'] < 50:
            moves.append(board.san(chess.Move.from_uci(move["Move"])))
    return moves

ds = ds.map(lambda x: {'best_move': get_top_moves(x["fen"])}, num_proc=54)
ds = ds.filter(lambda x: len(x['best_move']) > 0)
ds.save_to_disk("filtered_dataset_small_with_prompts_final")



