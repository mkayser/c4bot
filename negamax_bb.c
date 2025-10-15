// negamax_bb.c
// Bitboard Connect-4 (6x7). Layout: 7 bits per column (lowest is board bottom, highest is sentinel).
// Bit index = col*7 + row, where row ∈ [0..5] are cells, row=6 is sentinel.
// me/opp are disjoint; mask = me | opp.

#include "negamax_bb.h"
#include <stdint.h>   // uint64_t
#include <stdbool.h>  // bool
#include <limits.h>   // INT_MIN/INT_MAX
#include <assert.h>   // assert

// --- Compile-time board constants ---
enum {
    WIDTH  = 7,
    HEIGHT = 6,
    BITS_PER_COL = 7,
};

// Column mask helpers.
#define COL_MASK(c)     (UINT64_C(0x7F) << ((c) * BITS_PER_COL))     /* 7 bits in column c */
#define BOTTOM_MASK(c)  (UINT64_C(1)    << ((c) * BITS_PER_COL))     /* bottom cell of column c */
#define TOP_MASK(c)     (UINT64_C(1)    << ((c) * BITS_PER_COL + 6)) /* sentinel bit of column c */

// --- Public API (C ABI) ---
#ifdef __cplusplus
extern "C" {
#endif
int _best_move(uint64_t me, uint64_t opp, int depth);  /* Entry point: return column [0..6] */
int best_move(uint64_t me, uint64_t opp, int depth);   /* Alias for convenience/ABI stability */
#ifdef __cplusplus
}
#endif

// --- Core helpers (decls) ---
static inline uint64_t board_mask(uint64_t me, uint64_t opp);     /* me|opp */
static inline bool is_playable(uint64_t mask, int col);           /* !(mask & TOP_MASK(col)) */
static inline uint64_t drop(uint64_t player, uint64_t mask, int col); /* toggle the target bit into player */
static bool has_won(uint64_t bb);
static int generate_moves(uint64_t mask, int moves_out[WIDTH]);   /* returns count */
static int evaluate(uint64_t me, uint64_t opp);
static int negamax(uint64_t me, uint64_t opp, int depth, int alpha, int beta);

// --- Small utils needed by evaluate() ---
static inline int popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0; while (x) { x &= x - 1; ++c; } return c;
#endif
}

static inline int count_k(uint64_t bb, int d, int k) {
    uint64_t m = bb;
    for (int i = 1; i < k; ++i) m &= (bb >> (i * d));
    return popcnt64(m);
}

// --- Definitions ---

// Weights: tweak as you like.
#define W_WIN     100000
#define W_THREE       100
#define W_TWO          10
#define W_CENTER        3

static const int ORDER[7] = {3,4,2,5,1,6,0};  // center-first

int _best_move(uint64_t me, uint64_t opp, int depth) {
    uint64_t mask = board_mask(me, opp);
    int moves[WIDTH], n = generate_moves(mask, moves);
    if (n == 0) return -1;  // draw/no legal

    int best_col = moves[0];
    int alpha = -INT_MAX/2, beta = INT_MAX/2;
    int best_score = -INT_MAX/2;

    for (int i = 0; i < n; ++i) {
        int c = moves[i];
        uint64_t move = (mask + BOTTOM_MASK(c)) & COL_MASK(c);
        uint64_t me2  = me  ^ move;     // place

        // One-ply winning move shortcut.
        if (has_won(me2)) return c;

        int score = -negamax(opp, me2, depth - 1, -beta, -alpha);

        if (score > best_score) { best_score = score; best_col = c; }
        if (score > alpha) alpha = score;
    }
    return best_col;
}

int best_move(uint64_t me, uint64_t opp, int depth) {
    return _best_move(me, opp, depth);
}

static inline uint64_t board_mask(uint64_t me, uint64_t opp) {
    return me | opp;
}

static inline bool is_playable(uint64_t mask, int col) {
    assert(col >= 0 && col < WIDTH);
    return (mask & TOP_MASK(col)) == 0;
}

static inline uint64_t drop(uint64_t player, uint64_t mask, int col) {
    // Lowest empty bit in column col, toggle into player.
    uint64_t move = (mask + BOTTOM_MASK(col)) & COL_MASK(col);
    return player ^ move;   // XOR so unmake is symmetric
}

static bool has_won(uint64_t bb) {
    uint64_t m = bb & (bb >> 1);  if (m & (m >> 2)) return true;  // horizontal (1)
           m = bb & (bb >> 7);    if (m & (m >> 14)) return true; // vertical   (7)
           m = bb & (bb >> 6);    if (m & (m >> 12)) return true; // diag /     (6)
           m = bb & (bb >> 8);    if (m & (m >> 16)) return true; // diag \     (8)
    return false;
}

static int generate_moves(uint64_t mask, int out[7]) {
    int n = 0;
    for (int k = 0; k < 7; ++k) {
        int c = ORDER[k];
        if ((mask & TOP_MASK(c)) == 0) out[n++] = c;
    }
    return n;
}

static int evaluate(uint64_t me, uint64_t opp) {
    if (has_won(me))  return +W_WIN;
    if (has_won(opp)) return -W_WIN;

    const int H = 1, V = 7, D1 = 6, D2 = 8;

    int my3 = 0, my2 = 0;
    my3 += count_k(me, H, 3) + count_k(me, V, 3) + count_k(me, D1, 3) + count_k(me, D2, 3);
    my2 += count_k(me, H, 2) + count_k(me, V, 2) + count_k(me, D1, 2) + count_k(me, D2, 2);

    int op3 = 0, op2 = 0;
    op3 += count_k(opp, H, 3) + count_k(opp, V, 3) + count_k(opp, D1, 3) + count_k(opp, D2, 3);
    op2 += count_k(opp, H, 2) + count_k(opp, V, 2) + count_k(opp, D1, 2) + count_k(opp, D2, 2);

    uint64_t center_mask = COL_MASK(3) & ~TOP_MASK(3);
    int my_center  = popcnt64(me  & center_mask);
    int opp_center = popcnt64(opp & center_mask);

    int score = 0;
    score += W_THREE * (my3 - op3);
    score += W_TWO   * (my2 - op2);
    score += W_CENTER * (my_center - opp_center);

    return score;
}

static int negamax(uint64_t me, uint64_t opp, int depth, int alpha, int beta) {
    // If the previous player just made a connect-4, it's a loss for us.
    if (has_won(opp)) return -W_WIN;
    if (depth == 0)   return evaluate(me, opp);

    uint64_t mask = board_mask(me, opp);
    int moves[WIDTH], n = generate_moves(mask, moves);
    if (n == 0) return 0;  // draw

    int best = -INT_MAX/2;

    for (int i = 0; i < n; ++i) {
        int c = moves[i];
        uint64_t move = (mask + BOTTOM_MASK(c)) & COL_MASK(c);
        uint64_t me2  = me   ^ move;    // make
        uint64_t mask2= mask | move;
        (void)mask2; // mask2 is only needed to recompute child's moves; we rebuild from me2|opp below.

        // Immediate win shortcut.
        if (has_won(me2)) return W_WIN;

        int score = -negamax(opp, me2, depth - 1, -beta, -alpha);

        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;  // alpha-beta cutoff
    }
    return best;
}
