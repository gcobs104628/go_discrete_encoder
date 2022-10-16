#include <iostream>

// BitBoard format
// 8| 07 15 23 31 39 47 55 63
// 7| 06 14 22 30 38 46 54 62
// 6| 05 13 21 29 37 45 53 61
// 5| 04 12 20 28 36 44 52 60
// 4| 03 11 19 27 35 43 51 59
// 3| 02 10 18 26 34 42 50 58
// 2| 01 09 17 25 33 41 49 57
// 1| 00 08 16 24 32 40 48 56
//   一一一一一一一一一一一一一
//     a  b  c  d  e  f  g  h
// initial
//  ply1 = 0x0303030303030303
//  ply2 = 0xC0C0C0C0C0C0C0C0
//  N = 0x0081000000008100
//  B + Q = 0x0000810081810000
//  R + Q = 0x8100000081000081
//  P = 0x4242424242424242

namespace minizero::env::chess {

template <class T>
class ChessPair {
public:
    ChessPair() { reset(); }
    ChessPair(T white, T black) : white_(white), black_(black) {}
    inline void reset() { black_ = white_ = T(); }
    inline T& get(Player p) { return (p == Player::kPlayer1 ? white_ : black_); }
    inline const T& get(Player p) const { return (p == Player::kPlayer1 ? white_ : black_); }
    inline void set(Player p, const T& value) { (p == Player::kPlayer1 ? white_ : black_) = value; }
    inline void set(const T& white, const T& black)
    {
        white_ = white;
        black_ = black;
    }

private:
    T white_;
    T black_;
};

class Bitboard {
public:
    Bitboard(std::uint64_t board = 0) { setboard(board); }
    int count() const
    {
        std::uint64_t b = board_;
        b = (b & 0x5555555555555555) + ((b >> 1) & 0x5555555555555555);
        b = (b & 0x3333333333333333) + ((b >> 2) & 0x3333333333333333);
        return (((b + (b >> 4)) & 0x0f0f0f0f0f0f0f0f) * 0x0101010101010101) >> 56;
    }

    int countPawns() const
    {
        std::uint64_t b = board_;
        b &= 0x7E7E7E7E7E7E7E7E;
        b = (b & 0x5555555555555555) + ((b >> 1) & 0x5555555555555555);
        b = (b & 0x3333333333333333) + ((b >> 2) & 0x3333333333333333);
        return (((b + (b >> 4)) & 0x0f0f0f0f0f0f0f0f) * 0x0101010101010101) >> 56;
    }

    void showBitboard() const
    {
        for (int i = 0; i < 8; i++) {
            std::cout << 8 - i << "| ";
            for (int j = 0; j < 8; j++) {
                if (get(j * 8 + 7 - i))
                    std::cout << 1 << ' ';
                else
                    std::cout << 0 << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << "  一一一一一一一一\n";
        std::cout << "   a b c d e f g h\n";
    }
    void clear() { board_ = 0; }
    void setif(int square, bool cond) { board_ |= (std::uint64_t(cond) << square); }
    void set(int square) { board_ |= (std::uint64_t(1) << square); }
    void reset(int square) { board_ &= ~(std::uint64_t(1) << square); }
    void setboard(std::uint64_t board) { board_ = board; }
    bool get(int square) const { return board_ & (std::uint64_t(1) << square); }
    bool empty() { return board_ == 0; }
    bool intersects(const Bitboard& other) const { return board_ & other.board_; }
    bool operator==(const Bitboard& other) const { return board_ == other.board_; }
    bool operator!=(const Bitboard& other) const { return board_ != other.board_; }
    friend Bitboard operator|(Bitboard& a, const Bitboard& b) { return {a.board_ | b.board_}; }
    friend Bitboard operator&(Bitboard& a, const Bitboard& b) { return {a.board_ & b.board_}; }
    friend Bitboard operator-(Bitboard a, const Bitboard& b) { return {a.board_ & ~b.board_}; }

private:
    // 0~2^64-1
    std::uint64_t board_;
};

class Pieces_Bitboard {
public:
    ChessPair<Bitboard> pieces_;
    Bitboard pawns_;
    Bitboard knights_;
    Bitboard bishops_;
    Bitboard rooks_;
    Pieces_Bitboard() { reset(); }
    Pieces_Bitboard(ChessPair<Bitboard> pieces, Bitboard black, Bitboard pawn, Bitboard knight, Bitboard bishop, Bitboard rook)
        : pieces_(pieces), pawns_(pawn), knights_(knight), bishops_(bishop), rooks_(rook) {}
    void reset()
    {
        rooks_.setboard(0x8100000081000081);
        bishops_.setboard(0x0000810081810000);
        knights_.setboard(0x0081000000008100);
        pawns_.setboard(0x4242424242424242);
        Bitboard white(0x0303030303030303);
        Bitboard black(0xC0C0C0C0C0C0C0C0);
        pieces_.set(white, black);
    }
};

class Pieces_History {
public:
    std::vector<ChessPair<Bitboard>> pawn_;
    std::vector<ChessPair<Bitboard>> knight_;
    std::vector<ChessPair<Bitboard>> bishop_;
    std::vector<ChessPair<Bitboard>> rook_;
    std::vector<ChessPair<Bitboard>> queen_;
    std::vector<ChessPair<int>> king_;
    Pieces_History() { clear(); }
    void update(Pieces_Bitboard pieces, ChessPair<int> king_pos)
    {
        pawn_.push_back(ChessPair(pieces.pieces_.get(Player::kPlayer1) & pieces.pawns_, pieces.pieces_.get(Player::kPlayer2) & pieces.pawns_));
        knight_.push_back(ChessPair(pieces.pieces_.get(Player::kPlayer1) & pieces.knights_, pieces.pieces_.get(Player::kPlayer2) & pieces.knights_));
        bishop_.push_back(ChessPair((pieces.pieces_.get(Player::kPlayer1) & pieces.bishops_) - (pieces.bishops_ & pieces.rooks_), (pieces.pieces_.get(Player::kPlayer2) & pieces.bishops_) - (pieces.bishops_ & pieces.rooks_)));
        rook_.push_back(ChessPair((pieces.pieces_.get(Player::kPlayer1) & pieces.rooks_) - (pieces.bishops_ & pieces.rooks_), (pieces.pieces_.get(Player::kPlayer2) & pieces.rooks_) - (pieces.bishops_ & pieces.rooks_)));
        queen_.push_back(ChessPair(pieces.pieces_.get(Player::kPlayer1) & (pieces.bishops_ & pieces.rooks_), pieces.pieces_.get(Player::kPlayer2) & (pieces.bishops_ & pieces.rooks_)));
        king_.push_back(king_pos);
    }
    void clear()
    {
        pawn_.clear();
        knight_.clear();
        bishop_.clear();
        rook_.clear();
        queen_.clear();
        king_.clear();
    }
    int size() const
    {
        return pawn_.size();
    }
};

static const Bitboard kBishopAttacks[] = {
    0x8040201008040200ULL, 0x0080402010080500ULL, 0x0000804020110A00ULL,
    0x0000008041221400ULL, 0x0000000182442800ULL, 0x0000010204885000ULL,
    0x000102040810A000ULL, 0x0102040810204000ULL, 0x4020100804020002ULL,
    0x8040201008050005ULL, 0x00804020110A000AULL, 0x0000804122140014ULL,
    0x0000018244280028ULL, 0x0001020488500050ULL, 0x0102040810A000A0ULL,
    0x0204081020400040ULL, 0x2010080402000204ULL, 0x4020100805000508ULL,
    0x804020110A000A11ULL, 0x0080412214001422ULL, 0x0001824428002844ULL,
    0x0102048850005088ULL, 0x02040810A000A010ULL, 0x0408102040004020ULL,
    0x1008040200020408ULL, 0x2010080500050810ULL, 0x4020110A000A1120ULL,
    0x8041221400142241ULL, 0x0182442800284482ULL, 0x0204885000508804ULL,
    0x040810A000A01008ULL, 0x0810204000402010ULL, 0x0804020002040810ULL,
    0x1008050005081020ULL, 0x20110A000A112040ULL, 0x4122140014224180ULL,
    0x8244280028448201ULL, 0x0488500050880402ULL, 0x0810A000A0100804ULL,
    0x1020400040201008ULL, 0x0402000204081020ULL, 0x0805000508102040ULL,
    0x110A000A11204080ULL, 0x2214001422418000ULL, 0x4428002844820100ULL,
    0x8850005088040201ULL, 0x10A000A010080402ULL, 0x2040004020100804ULL,
    0x0200020408102040ULL, 0x0500050810204080ULL, 0x0A000A1120408000ULL,
    0x1400142241800000ULL, 0x2800284482010000ULL, 0x5000508804020100ULL,
    0xA000A01008040201ULL, 0x4000402010080402ULL, 0x0002040810204080ULL,
    0x0005081020408000ULL, 0x000A112040800000ULL, 0x0014224180000000ULL,
    0x0028448201000000ULL, 0x0050880402010000ULL, 0x00A0100804020100ULL,
    0x0040201008040201ULL};

static const Bitboard kKnightAttacks[] = {
    0x0000000000020400ULL, 0x0000000000050800ULL, 0x00000000000A1100ULL,
    0x0000000000142200ULL, 0x0000000000284400ULL, 0x0000000000508800ULL,
    0x0000000000A01000ULL, 0x0000000000402000ULL, 0x0000000002040004ULL,
    0x0000000005080008ULL, 0x000000000A110011ULL, 0x0000000014220022ULL,
    0x0000000028440044ULL, 0x0000000050880088ULL, 0x00000000A0100010ULL,
    0x0000000040200020ULL, 0x0000000204000402ULL, 0x0000000508000805ULL,
    0x0000000A1100110AULL, 0x0000001422002214ULL, 0x0000002844004428ULL,
    0x0000005088008850ULL, 0x000000A0100010A0ULL, 0x0000004020002040ULL,
    0x0000020400040200ULL, 0x0000050800080500ULL, 0x00000A1100110A00ULL,
    0x0000142200221400ULL, 0x0000284400442800ULL, 0x0000508800885000ULL,
    0x0000A0100010A000ULL, 0x0000402000204000ULL, 0x0002040004020000ULL,
    0x0005080008050000ULL, 0x000A1100110A0000ULL, 0x0014220022140000ULL,
    0x0028440044280000ULL, 0x0050880088500000ULL, 0x00A0100010A00000ULL,
    0x0040200020400000ULL, 0x0204000402000000ULL, 0x0508000805000000ULL,
    0x0A1100110A000000ULL, 0x1422002214000000ULL, 0x2844004428000000ULL,
    0x5088008850000000ULL, 0xA0100010A0000000ULL, 0x4020002040000000ULL,
    0x0400040200000000ULL, 0x0800080500000000ULL, 0x1100110A00000000ULL,
    0x2200221400000000ULL, 0x4400442800000000ULL, 0x8800885000000000ULL,
    0x100010A000000000ULL, 0x2000204000000000ULL, 0x0004020000000000ULL,
    0x0008050000000000ULL, 0x00110A0000000000ULL, 0x0022140000000000ULL,
    0x0044280000000000ULL, 0x0088500000000000ULL, 0x0010A00000000000ULL,
    0x0020400000000000ULL};

static const Bitboard kWhitePawnAttacks[] = {
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000200ULL,
    0x0000000000000400ULL, 0x0000000000000800ULL, 0x0000000000001000ULL,
    0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000020002ULL, 0x0000000000040004ULL,
    0x0000000000080008ULL, 0x0000000000100010ULL, 0x0000000000200020ULL,
    0x0000000000400040ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000002000200ULL, 0x0000000004000400ULL, 0x0000000008000800ULL,
    0x0000000010001000ULL, 0x0000000020002000ULL, 0x0000000040004000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000200020000ULL,
    0x0000000400040000ULL, 0x0000000800080000ULL, 0x0000001000100000ULL,
    0x0000002000200000ULL, 0x0000004000400000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000020002000000ULL, 0x0000040004000000ULL,
    0x0000080008000000ULL, 0x0000100010000000ULL, 0x0000200020000000ULL,
    0x0000400040000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0002000200000000ULL, 0x0004000400000000ULL, 0x0008000800000000ULL,
    0x0010001000000000ULL, 0x0020002000000000ULL, 0x0040004000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0200020000000000ULL,
    0x0400040000000000ULL, 0x0800080000000000ULL, 0x1000100000000000ULL,
    0x2000200000000000ULL, 0x4000400000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL,
    0x0008000000000000ULL, 0x0010000000000000ULL, 0x0020000000000000ULL,
    0x0040000000000000ULL};

static const Bitboard kBlackPawnAttacks[] = {
    0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
    0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000020002ULL,
    0x0000000000040004ULL, 0x0000000000080008ULL, 0x0000000000100010ULL,
    0x0000000000200020ULL, 0x0000000000400040ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000002000200ULL, 0x0000000004000400ULL,
    0x0000000008000800ULL, 0x0000000010001000ULL, 0x0000000020002000ULL,
    0x0000000040004000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000200020000ULL, 0x0000000400040000ULL, 0x0000000800080000ULL,
    0x0000001000100000ULL, 0x0000002000200000ULL, 0x0000004000400000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000020002000000ULL,
    0x0000040004000000ULL, 0x0000080008000000ULL, 0x0000100010000000ULL,
    0x0000200020000000ULL, 0x0000400040000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0002000200000000ULL, 0x0004000400000000ULL,
    0x0008000800000000ULL, 0x0010001000000000ULL, 0x0020002000000000ULL,
    0x0040004000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0200020000000000ULL, 0x0400040000000000ULL, 0x0800080000000000ULL,
    0x1000100000000000ULL, 0x2000200000000000ULL, 0x4000400000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0002000000000000ULL,
    0x0004000000000000ULL, 0x0008000000000000ULL, 0x0010000000000000ULL,
    0x0020000000000000ULL, 0x0040000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL};

static const Bitboard kRookAttacks[] = {
    0x01010101010101FEULL, 0x02020202020202FDULL, 0x04040404040404FBULL,
    0x08080808080808F7ULL, 0x10101010101010EFULL, 0x20202020202020DFULL,
    0x40404040404040BFULL, 0x808080808080807FULL, 0x010101010101FE01ULL,
    0x020202020202FD02ULL, 0x040404040404FB04ULL, 0x080808080808F708ULL,
    0x101010101010EF10ULL, 0x202020202020DF20ULL, 0x404040404040BF40ULL,
    0x8080808080807F80ULL, 0x0101010101FE0101ULL, 0x0202020202FD0202ULL,
    0x0404040404FB0404ULL, 0x0808080808F70808ULL, 0x1010101010EF1010ULL,
    0x2020202020DF2020ULL, 0x4040404040BF4040ULL, 0x80808080807F8080ULL,
    0x01010101FE010101ULL, 0x02020202FD020202ULL, 0x04040404FB040404ULL,
    0x08080808F7080808ULL, 0x10101010EF101010ULL, 0x20202020DF202020ULL,
    0x40404040BF404040ULL, 0x808080807F808080ULL, 0x010101FE01010101ULL,
    0x020202FD02020202ULL, 0x040404FB04040404ULL, 0x080808F708080808ULL,
    0x101010EF10101010ULL, 0x202020DF20202020ULL, 0x404040BF40404040ULL,
    0x8080807F80808080ULL, 0x0101FE0101010101ULL, 0x0202FD0202020202ULL,
    0x0404FB0404040404ULL, 0x0808F70808080808ULL, 0x1010EF1010101010ULL,
    0x2020DF2020202020ULL, 0x4040BF4040404040ULL, 0x80807F8080808080ULL,
    0x01FE010101010101ULL, 0x02FD020202020202ULL, 0x04FB040404040404ULL,
    0x08F7080808080808ULL, 0x10EF101010101010ULL, 0x20DF202020202020ULL,
    0x40BF404040404040ULL, 0x807F808080808080ULL, 0xFE01010101010101ULL,
    0xFD02020202020202ULL, 0xFB04040404040404ULL, 0xF708080808080808ULL,
    0xEF10101010101010ULL, 0xDF20202020202020ULL, 0xBF40404040404040ULL,
    0x7F80808080808080ULL};
} // namespace minizero::env::chess