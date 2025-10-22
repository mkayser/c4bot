#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "negamax_bb.h"

enum {
    WIDTH  = 7,
    HEIGHT = 6,
    BITS_PER_COL = 7,
};

#define COL_MASK(c)       (UINT64_C(0x7F) << ((c) * BITS_PER_COL))
#define BOTTOM_MASK(c)    (UINT64_C(1)    << ((c) * BITS_PER_COL))
#define TOP_CELL_MASK(c)  (UINT64_C(1)    << ((c) * BITS_PER_COL + (HEIGHT - 1)))

static inline uint64_t board_mask(uint64_t me, uint64_t opp) {
    return me | opp;
}

static bool __attribute__((unused))  has_won_ref(uint64_t bb) {
    uint64_t m = bb & (bb >> 1);  if (m & (m >> 2)) return true;
           m = bb & (bb >> 7);    if (m & (m >> 14)) return true;
           m = bb & (bb >> 6);    if (m & (m >> 12)) return true;
           m = bb & (bb >> 8);    if (m & (m >> 16)) return true;
    return false;
}


static bool apply_moves(const char *moves, uint64_t *me_out, uint64_t *opp_out) {
    uint64_t first = 0;
    uint64_t second = 0;
    bool first_turn = true;

    for (const char *p = moves; *p; ++p) {
        if (*p < '0' || *p > '6') {
            continue;
        }
        int col = *p - '0';
        uint64_t mask = board_mask(first, second);
        if (mask & TOP_CELL_MASK(col)) {
            fprintf(stderr, "Illegal move sequence '%s': column %d overflows\n", moves, col);
            return false;
        }
        uint64_t move = (mask + BOTTOM_MASK(col)) & COL_MASK(col);
        if (first_turn) {
            first |= move;
        } else {
            second |= move;
        }
        first_turn = !first_turn;
    }

    if (first_turn) {
        *me_out = first;
        *opp_out = second;
    } else {
        *me_out = second;
        *opp_out = first;
    }
    return true;
}

static void print_board(uint64_t me, uint64_t opp) {
    for (int row = HEIGHT - 1; row >= 0; --row) {
        printf("    ");
        for (int col = 0; col < WIDTH; ++col) {
            uint64_t bit = UINT64_C(1) << (col * BITS_PER_COL + row);
            char ch = '.';
            if (me & bit) {
                ch = 'X';
            } else if (opp & bit) {
                ch = 'O';
            }
            printf("%c", ch);
        }
        printf("\n");
    }
    printf("    0123456\n");
    printf("    (x is me)\n");
}

static bool __attribute__((unused)) is_move_legal(uint64_t mask, int col) {
    if (col < 0 || col >= WIDTH) {
        return false;
    }
    return (mask & TOP_CELL_MASK(col)) == 0;
}

int main(void) {

    Logger log = (Logger){ .f = stdout, .indent = 0, .verbosity = 2 };
    Logger* logp = &log;
    //Logger* logp = NULL;
    
    while(1) {
        printf("Enter a move sequence, depth to probe:  ");
        char moves[257];
        int depth;
        int field_count = scanf("%256s %d", moves, &depth);
        if (field_count == 2) {
            uint64_t me, opp;
            int n_moves = strlen(moves);
            if(!apply_moves(moves, &me, &opp)) {
                printf("Failed to perform move sequence\n");
                continue;
            }
            if(n_moves % 2 == 0) {
                // My turn
                printf("    My turn\n");
                printf("    Initial board:\n");
                print_board(me, opp);
                printf("currplayer: %#06lx\n", me);
                printf("curropp: %#06lx\n", opp);
                int result = best_move(me, opp, depth, logp);   
                printf("Best move: %d (one-based: %d) \n", result, result+1);
            }
            else {
                // Opp turn
                printf("    Opp turn\n");
                printf("    Initial board:\n");
                print_board(me, opp);
                printf("currplayer: %#06lx\n", me);
                printf("curropp: %#06lx\n", opp);
                int result = best_move(me, opp, depth, logp);   
                printf("Best move: %d\n", result);
            }
        }
        else {
            printf("Field count %d does not equal 2. Quitting...", field_count);
            break;
        }
    }
}
