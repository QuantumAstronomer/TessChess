import numpy as np
from BitboardHelpers import make_uint, set_bit, set_bit_multi, clear_bit, clear_bit_multi, bitboard_to_squares
from Attacks import knight_attack_maps, rook_attack_maps, bishop_attack_maps, queen_attack_maps, king_attack_maps, \
                    white_pawn_move_maps, white_pawn_attack_maps, black_pawn_move_maps, black_pawn_attack_maps
from Constants import File, Rank, HOT, Piece, Colour

class Board():
    """
    The board class will keep track of all the positions of each individual piece
    using a collection of bitboards. It also keeps track of overall board configurations
    like all occupied squares. Furthermore, the board class stores all movement and attack
    maps, these are dictionaries for each piece that contains which squares are attacked
    by a given piece on a given square.
    """

    def __init__(self):

        self.init_pieces()

        self.knight_attacks = knight_attack_maps()
        self.rook_attacks = rook_attack_maps()
        self.bishop_attacks = bishop_attack_maps()
        self.queen_attacks = queen_attack_maps()
        self.king_attacks = king_attack_maps()

        self.white_pawn_attack = white_pawn_attack_maps()
        self.white_pawn_move = white_pawn_move_maps()
        self.black_pawn_attack = black_pawn_attack_maps()
        self.black_pawn_move = black_pawn_move_maps()

    ## Board setup routine
    def init_pieces(self) -> None:

        ## Create and set all white piece bitboards
        self.white_pawn = make_uint()
        self.white_pawn |= set_bit_multi(self.white_pawn, range(8, 16))
        self.white_rook = make_uint()
        self.white_rook |= set_bit_multi(self.white_rook, [0, 7])
        self.white_knight = make_uint()
        self.white_knight |= set_bit_multi(self.white_knight, [1, 6])
        self.white_bishop = make_uint()
        self.white_bishop |= set_bit_multi(self.white_bishop, [2, 5])
        self.white_queen = make_uint()
        self.white_queen |= set_bit(self.white_queen, 3)
        self.white_king = make_uint()
        self.white_king |= set_bit(self.white_king, 4)

        ## Create and set all black piece bitboards
        self.black_pawn = make_uint()
        self.black_pawn |= set_bit_multi(self.black_pawn, range(48, 56))
        self.black_rook = make_uint()
        self.black_rook |= set_bit_multi(self.black_rook, [56, 63])
        self.black_knight = make_uint()
        self.black_knight |= set_bit_multi(self.black_knight, [57, 62])
        self.black_bishop = make_uint()
        self.black_bishop |= set_bit_multi(self.black_bishop, [58, 61])
        self.black_queen = make_uint()
        self.black_queen |= set_bit(self.black_queen, 59)
        self.black_king = make_uint()
        self.black_king |= set_bit(self.black_king, 60)

    def to_letterbox(self) -> np.array:

        letterbox = np.full(shape = (64), fill_value = "--", dtype = np.object_)
        letterbox[bitboard_to_squares(self.white_pawn)] = "wP"
        letterbox[bitboard_to_squares(self.white_rook)] = "wR"
        letterbox[bitboard_to_squares(self.white_knight)] = "wN"
        letterbox[bitboard_to_squares(self.white_bishop)] = "wB"
        letterbox[bitboard_to_squares(self.white_queen)] = "wQ"
        letterbox[bitboard_to_squares(self.white_king)] = "wK"
        letterbox[bitboard_to_squares(self.black_pawn)] = "bP"
        letterbox[bitboard_to_squares(self.black_rook)] = "bR"
        letterbox[bitboard_to_squares(self.black_knight)] = "bN"
        letterbox[bitboard_to_squares(self.black_bishop)] = "bB"
        letterbox[bitboard_to_squares(self.black_queen)] = "bQ"
        letterbox[bitboard_to_squares(self.black_king)] = "bK"

        return letterbox.reshape(8, 8)

    ## ---------------------------- ##
    ## Board access piece locations ##
    ## ---------------------------- ##

    @property
    def white_pieces(self):
        return self.white_king | self.white_queen | self.white_rook | self.white_knight | self.white_bishop | self.white_pawn

    @property
    def black_pieces(self):
        return self.black_king | self.black_queen | self.black_rook | self.black_knight | self.black_bishop | self.black_pawn

    @property
    def occupied_squares(self):
        return self.black_pieces | self.white_pieces

    @property
    def empty_squares(self):
        return ~self.occupied_squares

    @property
    def white_pawn_east_attacks(self):
        return (self.white_pawn << np.uint(9)) & ~np.uint64(File.hexA)

    @property
    def white_pawn_west_attacks(self):
        return (self.white_pawn << np.uint(7)) & ~np.uint64(File.hexH)

    @property
    def white_pawn_attacks(self):
        return self.white_pawn_east_attacks | self.white_pawn_west_attacks

    @property
    def black_pawn_east_attacks(self):
        return (self.black_pawn >> np.uint(7)) & ~np.uint64(File.hexA)

    @property
    def black_pawn_west_attacks(self):
        return (self.black_pawn >> np.uint(9)) & ~np.uint64(File.hexH)

    @property
    def black_pawn_attacks(self):
        return self.black_pawn_east_attacks | self.black_pawn_west_attacks

    ## ---------------------------- ##
    ## Update position bitboards    ##
    ## ---------------------------- ##

    def update_position_bitboards(self, change_map):
        """
        Given the map of pieces that has changed position, update the position
        bitboards accordingly. For efficiency only those that have changed position
        """

        for key, val in change_map.items():

            # White Pieces
            if key == Piece.wP:
                self.white_pawn = np.uint64(0)
                self.white_pawn |= set_bit_multi(self.white_pawn, list(val))

            elif key == Piece.wR:
                self.white_rook = np.uint64(0)
                self.white_rook |= set_bit_multi(self.white_rook, list(val))

            elif key == Piece.wN:
                self.white_knight = np.uint64(0)
                self.white_knight |= set_bit_multi(self.white_knight, list(val))

            elif key == Piece.wB:
                self.white_bishop = np.uint64(0)
                self.white_bishop |= set_bit_multi(self.white_bishop, list(val))

            elif key == Piece.wQ:
                self.white_queen = np.uint64(0)
                self.white_queen |= set_bit_multi(self.white_queen, list(val))

            elif key == Piece.wK:
                self.white_king = np.uint64(0)
                self.white_king |= set_bit_multi(self.white_king, list(val))

            # Black Pieces
            if key == Piece.bP:
                self.black_pawn = np.uint64(0)
                self.black_pawn |= set_bit_multi(self.black_pawn, list(val))

            elif key == Piece.bR:
                self.black_rook = np.uint64(0)
                self.black_rook |= set_bit_multi(self.black_rook, list(val))

            elif key == Piece.bN:
                self.black_knight = np.uint64(0)
                self.black_knight |= set_bit_multi(self.black_knight, list(val))

            elif key == Piece.bB:
                self.black_bishop = np.uint64(0)
                self.black_bishop |= set_bit_multi(self.black_bishop, list(val))

            elif key == Piece.bQ:
                self.black_queen = np.uint64(0)
                self.black_queen |= set_bit_multi(self.black_queen, list(val))

            elif key == Piece.bK:
                self.black_king = np.uint64(0)
                self.black_king |= set_bit_multi(self.black_king, list(val))

    ## ---------------- ##
    ## Piece movements  ##
    ## ---------------- ##

    ## Knight movement
    def get_knight_attack(self, square):
        return self.knight_attacks[square]

    ## Rook movement
    def get_rook_attack(self, square):
        return self.rook_attacks[square]

    ## Bishop movement
    def get_bishop_attack(self, square):
        return self.bishop_attacks[square]

    ## Queen movement
    def get_queen_attack(self, square):
        return self.queen_attacks[square]

    ## King movement
    def get_king_attack(self, square):
        return self.king_attacks[square]

    ## Pawn movement
    def get_pawn_move(self, colour, square):
        if colour is Colour.WHITE:
            return self.white_pawn_moves[square]
        return self.black_pawn_moves[square]

    def get_pawn_attack(self, colour, square):
        if colour is Colour.WHITE:
            return self.white_pawn_attacks[square]
        return self.black_pawn_attack_maps[square]
