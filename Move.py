from Constants import Piece, Colour

class Move():

    def __init__(self, piece, squares: tuple[int, int]):
        self.piece = piece
        if squares:
            self.from_square = squares[0]
            self.to_square = squares[1]
        else:
            self.from_square = None
            self.to_square = None

        self.is_capture = False
        self.is_en_passant = False
        self.is_castling = False
        self.is_promotion = False
        self.promote_to = None


    @property
    def colour(self):
        if self.piece in Piece.white_pieces:
            return Colour.WHITE
        return Colour.BLACK

class MoveResult():

    def __init__(self):
        self.is_checkmate = False
        self.is_king_check = False
        self.is_stalemate = False
        self.is_draw_claim_allowed = False
        self.is_illegal_move = False
        self.fen = ''
