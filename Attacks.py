"""
This file contains all important routines to generate static bitboards for
all piece attacks/moves.
"""


import numpy as np
from BitboardHelpers import *


from Constants import HOT, Square, File, Rank, Piece, DARK_SQUARES, LIGHT_SQUARES, BOARD_SIZE, BOARD_SQUARES

## ---------------------------------------- ##
## Horizontal, Vertical, and Diagonal rays  ##
## ---------------------------------------- ##

def south_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of south sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the south ray sliding attacks
        from_square: square index of the piece from which to generate
        the south sliding ray
    Return:
        bitboard of all southern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, from_square + np.array(range(0, -64, -8)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def north_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of north sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the north ray sliding attacks
        from_square: square index of the piece from which to generate
        the north sliding ray
    Return:
        bitboard of all northern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, from_square + np.array(range(0, 64, 8)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def west_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of west sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the west ray sliding attacks
        from_square: square index of the piece from which to generate
        the west sliding ray
    Return:
        bitboard of all western squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, from_square + np.array(range(0, - (from_square % 8) - 1, -1)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def east_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of east sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the east ray sliding attacks
        from_square: square index of the piece from which to generate
        the east sliding ray
    Return:
        bitboard of all eastern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, from_square + np.array(range(0, from_square % 8, 1)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def southeast_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of southeast sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the southeast ray sliding attacks
        from_square: square index of the piece from which to generate
        the southeast sliding ray
    Return:
        bitboard of all southeastern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, np.array(range(from_square, 0, -7)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def southwest_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of southwest sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the southwest ray sliding attacks
        from_square: square index of the piece from which to generate
        the southwest sliding ray
    Return:
        bitboard of all southwestern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, np.array(range(from_square, 0, -9)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def northwest_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of northwest sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the northwest ray sliding attacks
        from_square: square index of the piece from which to generate
        the northwest sliding ray
    Return:
        bitboard of all northwestern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, np.array(range(from_square, 63, 7)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def northeast_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Generate static bitboard of northeast sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        bitboard: the bitboard representing the northeast ray sliding attacks
        from_square: square index of the piece from which to generate
        the northeast sliding ray
    Return:
        bitboard of all northeastern squares attacked on an empty bitboard
    """
    original_square = from_square
    bitboard = set_bit_multi(bitboard, np.array(range(from_square, 64, 9)))
    bitboard = clear_bit(bitboard, original_square)
    return bitboard

def file_attack(from_square: int) -> np.uint64:
    """
    Generate static bitboard of north-south sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        from_square: square index of the piece from which to generate
        the north-south sliding ray
    Return:
        bitboard of all squares in a file attacked on an empty bitboard
    """
    bitboard = make_uint()
    attack_board = (north_ray(bitboard, from_square) |
                    south_ray(bitboard, from_square))
    return clear_bit(attack_board, from_square)

def rank_attack(from_square: int) -> np.uint64:
    """
    Generate static bitboard of east-west sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        from_square: square index of the piece from which to generate
        the east-west sliding ray
    Return:
        bitboard of all squares in a rank attacked on an empty bitboard
    """
    bitboard = make_uint()
    attack_board = (east_ray(bitboard, from_square) |
                    west_ray(bitboard, from_square))
    return clear_bit(attack_board, from_square)

def diagonal_attack(from_square: int) -> np.uint64:
    """
    Generate static bitboard of diagonal sliding piece attacked squares
    on an otherwise empty board
    Parameters:
        from_square: square index of the piece from which to generate
        the diagonal sliding ray
    Return:
        bitboard of all squares attacked by a diagonally moving piece
        on an empty bitboard
    """
    bitboard = make_uint()
    attack_board = (northeast_ray(bitboard, from_square) |
                    northwest_ray(bitboard, from_square) |
                    southeast_ray(bitboard, from_square) |
                    southwest_ray(bitboard, from_square))
    return clear_bit(attack_board, from_square)

## ------------------------ ##
## Knight attack pattern    ##
## ------------------------ ##

def generate_knight_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a knight
    Parameters:
        from_square: square from which the knight is attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a knight
    """
    bitboard = make_uint()
    for i in [6, 10, 15, 17, -6, -10, -15, -17]:
        bitboard |= set_bit(bitboard, from_square + i)
        ## Mask for wrapping around when near edges of the board
        if from_square in (File.B | File.A):
            bitboard &= ~(np.uint64(File.hexG | File.hexH))
        if from_square in (File.G | File.H):
            bitboard &= ~(np.uint64(File.hexA | File.hexB))

    return bitboard

## ------------------------ ##
## Rook attack pattern      ##
## ------------------------ ##

def generate_rook_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a rook
    Parameters:
        from_square: square from which the rook is attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a rook
    """
    return file_attack(from_square) | rank_attack(from_square)

## ------------------------ ##
## Bishop attack pattern    ##
## ------------------------ ##

def generate_bishop_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a bishop
    Parameters:
        from_square: square from which the bishop is attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a bishop
    """
    return diagonal_attack(from_square)

## ------------------------ ##
## Queen attack pattern     ##
## ------------------------ ##

def generate_queen_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a queen
    Parameters:
        from_square: square from which the queen is attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a queen
    """
    return diagonal_attack(from_square) | file_attack(from_square) | rank_attack(from_square)

## ------------------------ ##
## King attack pattern      ##
## ------------------------ ##

def generate_king_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a king
    Parameters:
        from_square: square from which the king is attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a king
    """
    bitboard = make_uint()
    ## North/South attacks do not need a guard cause these indices simply do not
    ## exist if they are needed
    for i in [-8, 8]:
        bitboard |= HOT << np.uint(from_square + i)
    ## East direction attacks, need to guard we are not in File H because you can
    ## not attack in File A in this case
    for i in [-7, 1, 9]:
        bitboard |= HOT << np.uint(from_square + i) & ~np.uint64(File.hexA)
    ## West direction attacks, need to guard we are not in File A because you can
    ## not attack in File H in this case
    for i in [-9, -1, 7]:
        bitboard |= HOT << np.uint(from_square + i) & ~np.uint64(File.hexH)
    return bitboard

## ---------------------------- ##
## Pawn attack/move patterns    ##
## ---------------------------- ##

def generate_white_pawn_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a white pawn
    Parameters:
        from_square: square from which the white pawns are attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a white pawn
    """
    bitboard = make_uint()
    ## Northeast direction, mask A File cause we cannot attack that
    bitboard |= HOT << np.uint64(from_square + 9) & ~np.uint64(File.hexA)
    ## Northwest direction, mask H File cause we cannot attack that
    bitboard |= HOT << np.uint64(from_square + 7) & ~np.uint64(File.hexH)
    return bitboard

def generate_black_pawn_attack_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares attacked by a black pawn
    Parameters:
        from_square: square from which the black pawns are attacking to generate
        a bitboard
    Return:
        bitboard of all attacked squares by a black pawn
    """
    bitboard = make_uint()
    ## Southeast direction, mask A File cause we cannot attack that
    bitboard |= HOT << np.uint64(from_square - 9) & ~np.uint64(File.hexA)
    ## Southwest direction, mask H File cause we cannot attack that
    bitboard |= HOT << np.uint64(from_square - 7) & ~np.uint64(File.hexH)
    return bitboard

def generate_white_pawn_move_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares to which a white pawn can move
    Parameters:
        from_square: starting square from which the white pawn is moving
    Return:
        bitboard of all squares a white pawn can move to
    """
    bitboard = make_uint()
    bitboard |= HOT << np.uint64(from_square + 8)
    if from_square in Rank.x2:
        bitboard |= HOT << np.uint64(from_square + 16)
    return bitboard

def generate_black_pawn_move_bitboard(from_square: int) -> np.uint64:
    """
    Generate a static bitboard of squares to which a black pawn can move
    Parameters:
        from_square: starting square from which the black pawn is moving
    Return:
        bitboard of all squares a black pawn can move to
    """
    bitboard = make_uint()
    bitboard |= HOT << np.uint64(from_square - 8)
    if from_square in Rank.x7:
        bitboard |= HOT << np.uint64(from_square - 16)
    return bitboard

## ---------------------------------------------------- ##
## Generating the maps of all attacks for a given piece ##
## ---------------------------------------------------- ##

def rank_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static rank attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_rank_attack_bitboard(i)
    return attack_map

def file_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static file attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_file_attack_bitboard(i)
    return attack_map

def diagonal_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static diagonal attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_diagonal_attack_bitboard(i)
    return attack_map

def knight_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static knight attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_knight_attack_bitboard(i)
    return attack_map

def rook_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static rook attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_rook_attack_bitboard(i)
    return attack_map

def bishop_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static bishop attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_bishop_attack_bitboard(i)
    return attack_map

def queen_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static queen attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_queen_attack_bitboard(i)
    return attack_map

def king_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static king attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_king_attack_bitboard(i)
    return attack_map

def white_pawn_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static white pawn attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_white_pawn_attack_bitboard(i)
    return attack_map

def black_pawn_attack_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static black pawn attack
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_black_pawn_attack_bitboard(i)
    return attack_map

def white_pawn_move_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static white pawn move
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_white_pawn_move_bitboard(i)
    return attack_map

def black_pawn_move_maps() -> dict:
    """
    Build a (static) python dictionary to represent the static black pawn move
    patterns on an otherwise empty bitboard
    Returns:
        dictionary of all attack bitboards
    """
    attack_map = {}
    for i in range(BOARD_SQUARES):
        attack_map[i] = generate_black_pawn_move_bitboard(i)
    return attack_map
