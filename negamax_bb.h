#ifndef NEGAMAX_BB_H
#define NEGAMAX_BB_H

#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
  #ifdef NEGAMAXBB_BUILD
    #define NEGAMAX_API __declspec(dllexport)
  #else
    #define NEGAMAX_API __declspec(dllimport)
  #endif
#else
  #define NEGAMAX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { FILE *f; int indent; int verbosity; } Logger;

// Returns best column [0..6], or -1 if no legal moves.
NEGAMAX_API int best_move_no_log(uint64_t me, uint64_t opp, int depth);
NEGAMAX_API int best_move(uint64_t me, uint64_t opp, int depth, Logger* log);
NEGAMAX_API int _best_move(uint64_t me, uint64_t opp, int depth, Logger* log); // alias
NEGAMAX_API void nb_log(Logger *l, const char *fmt, ...);


#ifdef __cplusplus
}
#endif
#endif
