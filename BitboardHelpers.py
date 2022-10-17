"""
This file contains all operations that are useful when working with bitboards
including scanning functions, manipulation functions, and conversion functions.
"""

import string
import numpy as np
import math

from Constants import Rank, File, LIGHT_SQUARES, DARK_SQUARES


def make_uint() -> np.uint64:
    """
    Returns:
        an unsigned 64 bit integer with value equal to zero
    """
    return np.uint64(0)


## ---------------------------------------- ##
##  Bit Querying/Manipulation Functions     ##
## ---------------------------------------- ##

def set_bit(bitboard: np.uint64, bit: int or np.uint64) -> np.uint64:
    """
    Turn the given bit in a bitboard to a HOT bit if not the case,
    else do nothing
    Parameters:
        bitboard: bitboard in which to set the given bit to HOT
        bit: the number of the bit to set to HOT
    Returns
        a bitboard in which the given bit is set to HOT
    """
    return bitboard | np.uint64(1) << np.uint64(bit)

def set_bit_multi(bitboard: np.uint64, bits: list or np.array) -> np.uint64:
    """
    Set multiple bits in a bitboard to HOT
    Parameters:
        bitboard: bitboard in which to set bits to HOT
        bits: list of the bitnumbers to become HOT bits
    Returns:
        a bitboard with the listed bits set to HOT
    """
    for bit in bits:
        bitboard = set_bit(bitboard, bit)
    return bitboard

def clear_bit(bitboard: np.uint64, bit: int or np.uint64) -> np.uint64:
    """
    Turn the given bit in a bitboard to a COLD bit if not the case,
    else do nothing
    Parameters:
        bitboard: bitboard in which to set the given bit to COLD
        bit: the number of the bit to set to COLD
    Returns
        a bitboard in which the given bit is set to COLD
    """
    return bitboard & ~(np.uint64(1) << np.uint64(bit))

def clear_bit_multi(bitboard: np.uint64, bits: list or np.array) -> np.uint64:
    """
    Set multiple bits in a bitboard to COLD
    Parameters:
        bitboard: bitboard in which to set bits to COLD
        bits: list of the bitnumbers to become COLD bits
    Returns:
        a bitboard with the listed bits set to COLD
    """
    for bit in bits:
        bitboard = clear_bit(bitboard, bit)
    return bitboard


def forward_bitscan(bitboard: np.uint64) -> int:
    """
    Find the least significant bit from the given bitboard
    Parameters:
        bitboad: bitboard to scan
    Returns:
        integer representing the significant bit
    """
    if not bitboard or (bitboard == np.uint64(0)):
        raise Exception("You can not scan an empty/non-existing bitboard...")

    return int(math.log2(bitboard & -bitboard))

def backward_bitscan(bitboard: np.uint64) -> int:
    """
    Find the position of the most significant bit (MSB) from the given bitboard
    Parameters:
        bitboard: bitboard to scan
    Returns:
        integer giving the position of the MSB
    """
    if not bitboard or (bitboard == np.uint64(0)):
        raise Exception("You can not scan an empty/non-existing bitboard...")

    bitboard |= bitboard >> np.uint64(1)
    bitboard |= bitboard >> np.uint64(2)
    bitboard |= bitboard >> np.uint64(4)
    bitboard |= bitboard >> np.uint64(8)
    bitboard |= bitboard >> np.uint64(16)
    bitboard |= bitboard >> np.uint64(32)
    bitboard += np.uint64(1)

    return int(math.log2(bitboard >> np.uint64(1)))



## -------------------------------- ##
##  Bitboard Conversion Functions   ##
## -------------------------------- ##

def bitboard_pprint(bitboard: np.uint64, board_size: int or np.uint64 = 8) -> None:
    """
    Pretty-prints the given bitboard as an 8x8 chess board
    Parameters:
        bitboard: the bitboard to pretty-print
    """
    bitboard_string = bitboard_to_string(bitboard)
    pretty_board = ''
    display_rank = board_size
    board = np.array([bitboard_string[i:i + board_size] for i in range(0, len(bitboard_string), board_size)])
    for i, row in enumerate(board):
        pretty_board += f'{display_rank} '
        display_rank -= 1
        for square in reversed(row):
            if int(square):
                pretty_board += ' ▓'
                continue
            pretty_board += ' ░'
        pretty_board += '\n'
    pretty_board += '  '
    for char in string.ascii_uppercase[:board_size]:
        pretty_board += f' {char}'
    print(pretty_board)


def bitboard_to_bytes(bitboard: np.uint64) -> bytes:
    """
    Convert a bitboard to a Python bytes object representation
    Parameters:
        bitboard: the bitboard to be turned into a bytes object
    Returns:
        the byte representation of the given bitboard
    """
    return bitboard.tobytes()

def bitboard_to_string(bitboard: np.uint64, board_size: int = 64) -> str:
    """
    Convert a bitboard into a binary string representation
    Parameters:
        bitboard: the bitboard to be represented as a string
        board_size: the number of squares on the bitboard
    Returns:
        a string representation of the provided bitboard
    """
    return format(bitboard, 'b').zfill(board_size)

def bitboard_to_squares(bitboard: np.uint64, board_size: int = 64) -> list:
    """
    Return the squares that are occupied in a given bitboard
    Parameters:
        bitboard: the bitboard to find the occupied squares in
        board_size: the number of squars on the bitboard
    Returns:
        a list of the occupied squares in a given bitboard
    """
    binary = bitboard_to_string(bitboard)
    squares = []
    for i, bit in enumerate(reversed(binary)):
        if int(bit):
            squares.append(i)
    return squares

## -------------------------------- ##
## Bitboard access of board regions ##
## -------------------------------- ##

## Rank access
def rank1() -> np.uint64:
    return np.uint64(Rank.hex1)

def rank2() -> np.uint64:
    return np.uint64(Rank.hex1)

def rank3() -> np.uint64:
    return np.uint64(Rank.hex3)

def rank4() -> np.uint64:
    return np.uint64(Rank.hex4)

def rank5() -> np.uint64:
    return np.uint64(Rank.hex5)

def rank6() -> np.uint64:
    return np.uint64(Rank.hex6)

def rank7() -> np.uint64:
    return np.uint64(Rank.hex7)

def rank8() -> np.uint64:
    return np.uint64(Rank.hex8)

## File access
def fileA() -> np.uint64:
    return np.uint64(File.hexA)

def fileB() -> np.uint64:
    return np.uint64(File.hexB)

def fileC() -> np.uint64:
    return np.uint64(File.hexC)

def fileD() -> np.uint64:
    return np.uint64(File.hexD)

def fileE() -> np.uint64:
    return np.uint64(File.hexE)

def fileF() -> np.uint64:
    return np.uint64(File.hexF)

def fileG() -> np.uint64:
    return np.uint64(File.hexG)

def fileH() -> np.uint64:
    return np.uint64(File.hexH)

## Colour access
def dark_suares() -> np.uint64:
    return np.uint64(DARK_SQUARES)

def light_squares() -> np.uint64:
    return np.uint64(LIGHT_SQUARES)

## Board regions
def center() -> np.uint64:
    return (fileD() | fileE()) & (rank3() | rank4())

def flanks() -> np.uint64:
    return fileA() | fileH()

def center_files() -> np.uint64:
    return fileC() | fileD() | fileE() | fileF()

def kingside() -> np.uint64:
    return fileE() | fileF() | fileG() | fileH()

def queenside() -> np.uint64:
    return fileA() | fileB() | fileC() | fileD()

def blackside() -> np.uint64:
    return rank5() | rank6() | rank7() | rank8()

def whiteside() -> np.uint64:
    return rank1() | rank2() | rank3() | rank4()
