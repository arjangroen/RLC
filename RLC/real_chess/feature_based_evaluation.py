from real_chess.environment import Board
from chess import WHITE, BLACK


def material_balance_score(board: Board):
    return board.get_material_value()


def movement_balance_score(board: Board):
    current_player = len([x for x in board.board.generate_legal_moves()])

    # Assume move balance from previous situation for opponent
    prev_move = board.board.pop()
    other_player = len([x for x in board.board.generate_legal_moves()])
    board.board.push(prev_move)

    if board.board.turn:
        return current_player - other_player
    else:
        return other_player - current_player


def is_check(board: Board):
    return board.board.is_check()


def castling_rights_score(board: Board):
    return board.board.has_castling_rights(WHITE) - board.board.has_castling_rights(BLACK)
