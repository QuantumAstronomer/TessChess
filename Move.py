from Constants import Piece, Colour

class Move():

    def __init__(self, piece = None, squares = None):
        self.piece = piece
        if squares:
            self.from = squares[0]
            self.to = squares[1]
        else:
            self.from = None
            self.to = None

        self.is_capture = False
        self.is_en_passant = False
        self.is_castle = False
        self.is_promotion = False
        self.promote_to = None


    @propety
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
