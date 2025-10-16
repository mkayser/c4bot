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

static bool has_won_ref(uint64_t bb) {
    uint64_t m = bb & (bb >> 1);  if (m & (m >> 2)) return true;
           m = bb & (bb >> 7);    if (m & (m >> 14)) return true;
           m = bb & (bb >> 6);    if (m & (m >> 12)) return true;
           m = bb & (bb >> 8);    if (m & (m >> 16)) return true;
    return false;
}

static int generate_moves(uint64_t mask, int moves_out[WIDTH]) {
    static const int ORDER[WIDTH] = {3, 4, 2, 5, 1, 6, 0};
    int n = 0;
    for (int k = 0; k < WIDTH; ++k) {
        int c = ORDER[k];
        if ((mask & TOP_CELL_MASK(c)) == 0) {
            moves_out[n++] = c;
        }
    }
    return n;
}

static int ref_negamax(uint64_t me, uint64_t opp, int depth) {
    if (has_won_ref(opp)) {
        return -1;
    }
    if (depth == 0) {
        return 0;
    }

    uint64_t mask = board_mask(me, opp);
    int moves[WIDTH];
    int n = generate_moves(mask, moves);
    if (n == 0) {
        return 0;
    }

    int best = -2;
    for (int i = 0; i < n; ++i) {
        int c = moves[i];
        uint64_t move = (mask + BOTTOM_MASK(c)) & COL_MASK(c);
        uint64_t child_me = me | move;

        if (has_won_ref(child_me)) {
            return 1;
        }

        int score = -ref_negamax(opp, child_me, depth - 1);
        if (score > best) {
            best = score;
        }
        if (best == 1) {
            break;
        }
    }

    if (best == -2) {
        best = 0;
    }
    return best;
}

static int reference_best_move(uint64_t me, uint64_t opp, int depth) {
    uint64_t mask = board_mask(me, opp);
    int moves[WIDTH];
    int n = generate_moves(mask, moves);
    if (n == 0) {
        return -1;
    }

    int best_col = moves[0];
    int best_score = -3;
    for (int i = 0; i < n; ++i) {
        int c = moves[i];
        uint64_t move = (mask + BOTTOM_MASK(c)) & COL_MASK(c);
        uint64_t child_me = me | move;
        int score;
        if (has_won_ref(child_me)) {
            score = 1;
        } else {
            score = -ref_negamax(opp, child_me, depth - 1);
        }

        if (score > best_score) {
            best_score = score;
            best_col = c;
        }
    }
    return best_col;
}

typedef struct {
    const char *name;
    const char *moves;
    int depth;
} TestCase;

typedef struct {
    uint64_t first;
    uint64_t second;
    bool first_turn;
} RawState;

typedef struct {
    char sequence[64];
    int depth;
    int expected;
    int got;
    uint64_t me;
    uint64_t opp;
} Mismatch;

typedef struct {
    size_t states_checked;
    size_t mismatches;
    size_t max_states;      /* 0 => unlimited */
    size_t max_recorded;    /* how many mismatches to store for reporting */
    size_t recorded;
    Mismatch records[32];
} SearchStats;

typedef struct {
    const char *name;
    int depth;
    int max_plies;
    size_t max_states;
    size_t max_reported;
} SearchSpec;

static bool apply_moves(const char *moves, uint64_t *me_out, uint64_t *opp_out) {
    uint64_t first = 0;
    uint64_t second = 0;
    bool first_turn = true;

    for (const char *p = moves; *p; ++p) {
        if (*p < '1' || *p > '7') {
            continue;
        }
        int col = *p - '1';
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
    printf("    1234567\n");
}

static void record_mismatch(SearchStats *stats,
                            const char *sequence,
                            size_t seq_len,
                            int depth,
                            int expected,
                            int got,
                            uint64_t me,
                            uint64_t opp) {
    ++stats->mismatches;
    if (stats->recorded >= stats->max_recorded) {
        return;
    }

    Mismatch *m = &stats->records[stats->recorded++];
    if (seq_len == 0) {
        strcpy(m->sequence, "(empty)");
    } else {
        size_t capped = seq_len;
        if (capped >= sizeof(m->sequence)) {
            capped = sizeof(m->sequence) - 1;
        }
        memcpy(m->sequence, sequence, capped);
        m->sequence[capped] = '\0';
    }
    m->depth = depth;
    m->expected = expected;
    m->got = got;
    m->me = me;
    m->opp = opp;
}

static void dfs_search(SearchStats *stats,
                       RawState state,
                       size_t seq_len,
                       int depth,
                       int max_plies,
                       char *sequence) {
    if (stats->max_states && stats->states_checked >= stats->max_states) {
        return;
    }
    if (stats->recorded >= stats->max_recorded) {
        return;
    }

    uint64_t me = state.first_turn ? state.first : state.second;
    uint64_t opp = state.first_turn ? state.second : state.first;

    if (has_won_ref(me) || has_won_ref(opp)) {
        return;  // game already over, no need to analyse further
    }

    uint64_t mask = board_mask(me, opp);
    int moves[WIDTH];
    int n = generate_moves(mask, moves);
    if (n == 0) {
        return;  // draw position
    }

    ++stats->states_checked;

    int expected = reference_best_move(me, opp, depth);
    int got = best_move(me, opp, depth, NULL);
    if (expected != got) {
        record_mismatch(stats, sequence, seq_len, depth, expected, got, me, opp);
    }

    if ((int)seq_len >= max_plies) {
        return;
    }

    for (int i = 0; i < n; ++i) {
        int col = moves[i];
        uint64_t move = (mask + BOTTOM_MASK(col)) & COL_MASK(col);

        RawState child = state;
        bool win;
        if (state.first_turn) {
            child.first ^= move;
            win = has_won_ref(child.first);
        } else {
            child.second ^= move;
            win = has_won_ref(child.second);
        }
        child.first_turn = !state.first_turn;

        sequence[seq_len] = (char)('1' + col);
        sequence[seq_len + 1] = '\0';

        if (!win) {
            dfs_search(stats, child, seq_len + 1, depth, max_plies, sequence);
        }

        sequence[seq_len] = '\0';

        if ((stats->max_states && stats->states_checked >= stats->max_states) ||
            stats->recorded >= stats->max_recorded) {
            return;
        }
    }
}

static bool run_search_spec(const SearchSpec *spec) {
    SearchStats stats = {0};
    stats.max_states = spec->max_states;
    stats.max_recorded = spec->max_reported;

    RawState root = {0, 0, true};
    char sequence[sizeof(((Mismatch *)0)->sequence)] = "";

    dfs_search(&stats, root, 0, spec->depth, spec->max_plies, sequence);

    printf("[%s] depth=%d, checked=%zu, mismatches=%zu\n",
           spec->name, spec->depth, stats.states_checked, stats.mismatches);

    for (size_t i = 0; i < stats.recorded; ++i) {
        const Mismatch *m = &stats.records[i];
        printf("  Sequence %s (depth=%d): reference=%d (col %d) vs library=%d (col %d)\n",
               m->sequence, m->depth,
               m->expected, (m->expected >= 0) ? (m->expected + 1) : -1,
               m->got, (m->got >= 0) ? (m->got + 1) : -1);
        printf("  me=0x%016" PRIx64 ", opp=0x%016" PRIx64 "\n", m->me, m->opp);
        print_board(m->me, m->opp);
    }

    return stats.mismatches == 0;
}

static bool is_move_legal(uint64_t mask, int col) {
    if (col < 0 || col >= WIDTH) {
        return false;
    }
    return (mask & TOP_CELL_MASK(col)) == 0;
}

int main(void) {
    const TestCase tests[] = {
        {"Immediate vertical win", "121212", 1},
        {"Immediate horizontal win", "334455", 1},
        {"Simple blocking move", "1212333", 2},
        {"Simple double threat", "2233", 2},
        {"Avoid simple threat", "454445454111111222222", 2}
    };
    const size_t num_tests = sizeof(tests) / sizeof(tests[0]);

    size_t failures = 0;
    for (size_t i = 0; i < num_tests; ++i) {
        const TestCase *tc = &tests[i];
        uint64_t me, opp;
        if (!apply_moves(tc->moves, &me, &opp)) {
            ++failures;
            continue;
        }
        int expected = reference_best_move(me, opp, tc->depth);
        int got = best_move(me, opp, tc->depth, NULL);

        printf("[%zu/%zu] %s: reference=%d, library=%d\n",
               i + 1, num_tests, tc->name, expected, got);

        uint64_t mask = board_mask(me, opp);
        int moves[WIDTH];
        int move_count = generate_moves(mask, moves);
        for (int j = 0; j < move_count; ++j) {
            if (!is_move_legal(mask, moves[j])) {
                fprintf(stderr,
                        "  ERROR: generate_moves produced illegal column %d for mask=0x%016" PRIx64 "\n",
                        moves[j], mask);
                ++failures;
            }
        }

        bool ref_in_moves = false;
        bool lib_in_moves = false;
        for (int j = 0; j < move_count; ++j) {
            if (moves[j] == expected) {
                ref_in_moves = true;
            }
            if (moves[j] == got) {
                lib_in_moves = true;
            }
        }

        if (expected >= 0 && !is_move_legal(mask, expected)) {
            fprintf(stderr,
                    "  ERROR: reference move %d (col %d) is illegal for mask=0x%016" PRIx64 "\n",
                    expected, expected + 1, mask);
            ++failures;
        } else if (expected >= 0 && !ref_in_moves) {
            fprintf(stderr,
                    "  ERROR: reference move %d (col %d) not present in generated moves for mask=0x%016" PRIx64 "\n",
                    expected, expected + 1, mask);
            ++failures;
        }
        if (got >= 0 && !is_move_legal(mask, got)) {
            fprintf(stderr,
                    "  ERROR: library move %d (col %d) is illegal for mask=0x%016" PRIx64 "\n",
                    got, got + 1, mask);
            ++failures;
        } else if (got >= 0 && !lib_in_moves) {
            fprintf(stderr,
                    "  ERROR: library move %d (col %d) not present in generated moves for mask=0x%016" PRIx64 "\n",
                    got, got + 1, mask);
            ++failures;
        }
        if (expected != got) {
            ++failures;
        }
    }

    const SearchSpec specs[] = {
        {"Full-search prefixes", 2, 7, 0, 8},
    };

    const size_t num_specs = sizeof(specs) / sizeof(specs[0]);
    const size_t total_groups = num_tests + num_specs;
    for (size_t i = 0; i < num_specs; ++i) {
        const SearchSpec *spec = &specs[i];
        bool ok = run_search_spec(spec);
        if (!ok) {
            ++failures;
        }
    }

    if (failures) {
        fprintf(stderr, "FAILED: %zu/%zu checks failed\n", failures, total_groups);
        return EXIT_FAILURE;
    }

    printf("All tests passed (%zu).", num_tests);
    if (num_specs) {
        printf(" Search suites run: %zu.\n", num_specs);
    } else {
        printf("\n");
    }
    return EXIT_SUCCESS;
}
