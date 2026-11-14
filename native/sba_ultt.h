/*
 * sba_ultt.h — shared board definitions for the SBA native engine (D27).
 *
 * State encoding (all engines agree on this):
 *   cells[81] : 0 empty, 1 = X, 2 = O          (macro m, cell i -> m*9+i)
 *   macro[9]  : 0 open/undecided, 1 X, 2 O, 3 full-draw
 *   player    : 1 X, 2 O (side to move)
 *   active    : confined macro index, or -1 = any open macro
 */

#ifndef SBA_ULTT_H
#define SBA_ULTT_H

#include <stdint.h>

#define SBA_N_MACROS 9

extern const int32_t SBA_LINES[8][3];

typedef struct {
    int32_t cells[81];
    int32_t macro[SBA_N_MACROS];
    int32_t player;
    int32_t active;
} SbaState;

/* Search switches that used to be Python-only (D37). Passing NULL to a
 * backend selects the conservative defaults (mirrors ai.py _CFG_DEFAULTS). */
typedef struct {
    double uct_c;              /* UCT exploration constant */
    int32_t dynamic_uct;       /* 1: anneal c as the search matures */
    int32_t rollout_heuristic; /* 1: win/block biased playout, 0: uniform */
    int32_t micro_tablebase;   /* 1: exact 3x3 micro values in minimax eval */
    int32_t symmetry;          /* 1: D4 root dedup on canonical positions */
    int32_t use_pattern;       /* 1: pattern rollout kernel (D26) */
} SbaSearchOpts;

/* Opaque, C-owned search tree. The arena is malloc'd inside the DLL, so no
 * uint8_t* -> MctsNode* cast (and no alignment assumption) crosses ctypes. */
typedef void *SbaTreeHandle;

/* rules */
int32_t sba_wins_micro(const int32_t *cells, int32_t m, int32_t i,
                       int32_t player);
int32_t sba_macro_line(const int32_t *macro, int32_t player);
int32_t sba_apply(SbaState *st, int32_t m, int32_t i);
/* returns 1 X / 2 O / 3 draw when terminal, 0 otherwise */
int32_t sba_result(const SbaState *st);

/* rng */
uint64_t sba_rng_next(uint64_t *s);
uint32_t sba_rng_below(uint64_t *s, uint32_t n);

/* rollouts (sba_rollout.c) */
int32_t __stdcall sba_rollout(const int32_t *cells, const int32_t *macro,
                              int32_t player, int32_t active, uint64_t seed);
int32_t __stdcall sba_rollout_pattern(const int32_t *cells, const int32_t *macro,
                                      int32_t player, int32_t active,
                                      uint64_t seed);
/* uniform-random playout (rollout_heuristic = false) */
int32_t __stdcall sba_rollout_random(const int32_t *cells, const int32_t *macro,
                                     int32_t player, int32_t active,
                                     uint64_t seed);
int32_t __stdcall sba_rollout_trace(const int32_t *cells, const int32_t *macro,
                                    int32_t player, int32_t active, uint64_t seed,
                                    int32_t *out_moves, int32_t *out_len,
                                    const SbaSearchOpts *opts);

/* full MCTS (sba_mcts.c) — D37: C owns the arena and hands back an opaque
 * tree handle; sba_tree_free() releases it. prev_tree may be NULL. */
int32_t __stdcall sba_tree_free(SbaTreeHandle tree);
int32_t __stdcall sba_mcts_search(const int32_t *cells, const int32_t *macro,
                                  int32_t player, int32_t active,
                                  int32_t iterations, uint64_t seed,
                                  const SbaSearchOpts *opts,
                                  int32_t *out_best, int32_t *out_count,
                                  int32_t *out_moves, double *out_visits,
                                  double *out_wins, int32_t *out_mover,
                                  int32_t *out_draws,
                                  SbaTreeHandle *out_tree,
                                  const SbaTreeHandle prev_tree);

/* minimax family (sba_minimax.c) */
int32_t __stdcall sba_minimax_normal(const int32_t *board9, int32_t player,
                                     int32_t *out_best, double *out_score);
int32_t __stdcall sba_minimax_ultimate(const int32_t *cells,
                                       const int32_t *macro, int32_t player,
                                       int32_t active, int32_t depth,
                                       const SbaSearchOpts *opts,
                                       int32_t *out_best);
int32_t __stdcall sba_flat_mcts(const int32_t *cells, const int32_t *macro,
                                int32_t player, int32_t active,
                                int32_t per_move, uint64_t seed,
                                const SbaSearchOpts *opts, int32_t *out_best);

#endif /* SBA_ULTT_H */
