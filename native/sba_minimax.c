/*
 * sba_minimax.c — Minimax family backends in C (D27):
 *   - sba_minimax_normal : full alpha-beta solve of 3x3 (negamax form)
 *   - sba_minimax_ultimate : depth-limited alpha-beta with the heuristic
 *     eval (fork / threat / occupancy / macro-pair terms — the Python
 *     micro-tablebase term is not carried over; near the leaves the search
 *     itself resolves those positions)
 *   - sba_flat_mcts : root-level independent playouts per legal move
 */

#include "sba_ultt.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/* 3x3 normal board: 9 cells, 1 X / 2 O / 0 empty                      */
/* ------------------------------------------------------------------ */

static int32_t normal_winner(const int32_t *b) {
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = SBA_LINES[l];
        if (b[ln[0]] != 0 && b[ln[0]] == b[ln[1]] && b[ln[1]] == b[ln[2]]) {
            return b[ln[0]];
        }
    }
    return 0;
}

/* negamax value from the perspective of the side to move; 10 = win,
 * 0 = draw, -10 = loss (depth-adjusted on the caller side) */
static int32_t negamax_normal(int32_t *b, int32_t player, int32_t alpha,
                              int32_t beta) {
    int32_t w = normal_winner(b);
    if (w != 0) return -10;              /* previous mover just won */
    int32_t any = 0;
    for (int32_t i = 0; i < 9; i++) {
        if (b[i] == 0) { any = 1; break; }
    }
    if (!any) return 0;

    int32_t best = -100;
    int32_t opp = player == 1 ? 2 : 1;
    for (int32_t i = 0; i < 9; i++) {
        if (b[i] != 0) continue;
        b[i] = player;
        int32_t v = -negamax_normal(b, opp, -beta, -alpha);
        b[i] = 0;
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (alpha >= beta) break;
    }
    return best;
}

int32_t __stdcall sba_minimax_normal(const int32_t *board9, int32_t player,
                                     int32_t *out_best, double *out_score) {
    int32_t b[9];
    for (int32_t i = 0; i < 9; i++) b[i] = board9[i];
    int32_t opp = player == 1 ? 2 : 1;
    /* center first, then corners/edges — matches the Python tie-break order */
    static const int32_t ORDER[9] = {4, 0, 2, 6, 8, 1, 3, 5, 7};
    int32_t best_score = -1000;
    int32_t best_move = -1;
    for (int32_t k = 0; k < 9; k++) {
        int32_t i = ORDER[k];
        if (b[i] != 0) continue;
        b[i] = player;
        int32_t w = normal_winner(b);
        int32_t v;
        if (w == player) v = 10;
        else if (w == opp) v = -10;      /* unreachable */
        else v = -negamax_normal(b, opp, -1000, 1000);
        b[i] = 0;
        if (v > best_score) {
            best_score = v;
            best_move = i;
        }
    }
    if (out_best) *out_best = best_move;
    if (out_score) *out_score = (double)best_score;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Ultimate: heuristic eval + alpha-beta                               */
/* ------------------------------------------------------------------ */

static int32_t count_threats(const int32_t *cells, int32_t m, int32_t player) {
    int32_t n = 0;
    const int32_t *c = cells + m * 9;
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = SBA_LINES[l];
        int32_t mine = 0, empty = 0, other = 0;
        for (int32_t k = 0; k < 3; k++) {
            int32_t v = c[ln[k]];
            if (v == player) mine++;
            else if (v == 0) empty++;
            else other++;
        }
        if (mine == 2 && empty == 1 && other == 0) n++;
    }
    return n;
}

static int32_t fork_count(const int32_t *cells, int32_t m, int32_t player) {
    /* empty cells whose play creates >=2 winning lines (Python _fork_count) */
    int32_t n = 0;
    for (int32_t i = 0; i < 9; i++) {
        if (cells[m * 9 + i] != 0) continue;
        if (sba_wins_micro(cells, m, i, player)) {
            /* a winning cell creates at least 1 line; count as fork when it
             * completes 2+ lines — sba_wins_micro returns 1 for any line;
             * approximate by counting the lines through this cell */
            int32_t lines = 0;
            for (int32_t l = 0; l < 8; l++) {
                const int32_t *ln = SBA_LINES[l];
                int32_t hit = 1;
                for (int32_t k = 0; k < 3; k++) {
                    int32_t idx = ln[k];
                    if (idx == i) continue;
                    if (cells[m * 9 + idx] != player) { hit = 0; break; }
                }
                lines += hit;
            }
            if (lines >= 2) n++;
        }
    }
    return n;
}

/* heuristic score from `player`'s perspective — mirrors eval_ultimate minus
 * the micro-tablebase term */
static double eval_ultt(const SbaState *st, int32_t player) {
    int32_t opp = player == 1 ? 2 : 1;
    double score = 0.0;
    for (int32_t m = 0; m < SBA_N_MACROS; m++) {
        int32_t state = st->macro[m];
        if (state == player) { score += 1000; continue; }
        if (state == opp) { score -= 1000; continue; }
        if (state != 0) continue;    /* full-draw macro */
        const int32_t *c = st->cells + m * 9;
        int32_t mine = 0, theirs = 0, empty = 0;
        for (int32_t i = 0; i < 9; i++) {
            if (c[i] == player) mine++;
            else if (c[i] == opp) theirs++;
            else empty++;
        }
        score += 40.0 * fork_count(st->cells, m, player);
        score -= 40.0 * fork_count(st->cells, m, opp);
        score += 3.0 * count_threats(st->cells, m, player);
        score -= 3.0 * count_threats(st->cells, m, opp);
        score += 0.5 * mine;
        score -= 0.5 * theirs;
    }
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = SBA_LINES[l];
        int32_t mine = 0, theirs = 0, empty = 0;
        for (int32_t k = 0; k < 3; k++) {
            int32_t v = st->macro[ln[k]];
            if (v == player) mine++;
            else if (v == opp) theirs++;
            else if (v == 0) empty++;
        }
        if (mine == 2 && empty == 1) score += 20;
        if (theirs == 2 && empty == 1) score -= 20;
    }
    return score;
}

static int32_t legal_list(const SbaState *st, int32_t *moves) {
    int32_t n = 0;
    for (int32_t m = 0; m < SBA_N_MACROS; m++) {
        if (st->active >= 0 && m != st->active) continue;
        if (st->macro[m] != 0) continue;
        for (int32_t i = 0; i < 9; i++) {
            if (st->cells[m * 9 + i] == 0) moves[n++] = m * 9 + i;
        }
    }
    return n;
}

static double ab_ultt(SbaState *st, int32_t depth, double alpha, double beta,
                      int32_t maximizing, int32_t ai_player) {
    int32_t result = sba_result(st);
    if (result != 0) {
        if (result == 3) return 0.0;
        return result == ai_player ? 100000.0 - depth : depth - 100000.0;
    }
    if (depth == 0) return eval_ultt(st, ai_player);

    int32_t moves[81];
    int32_t n = legal_list(st, moves);
    if (maximizing) {
        double best = -1e300;
        for (int32_t k = 0; k < n; k++) {
            int32_t m = moves[k] / 9, i = moves[k] % 9;
            int32_t saved_macro = st->macro[m];
            int32_t saved_active = st->active;
            int32_t saved_player = st->player;
            sba_apply(st, m, i);
            double v = ab_ultt(st, depth - 1, alpha, beta, 0, ai_player);
            st->macro[m] = saved_macro;
            st->active = saved_active;   /* sba_apply flips player too */
            st->player = saved_player;
            st->cells[m * 9 + i] = 0;    /* undo the placement */
            if (v > best) best = v;
            if (v > alpha) alpha = v;
            if (beta <= alpha) break;
        }
        return best;
    }
    double best = 1e300;
    for (int32_t k = 0; k < n; k++) {
        int32_t m = moves[k] / 9, i = moves[k] % 9;
        int32_t saved_macro = st->macro[m];
        int32_t saved_active = st->active;
        int32_t saved_player = st->player;
        sba_apply(st, m, i);
        double v = ab_ultt(st, depth - 1, alpha, beta, 1, ai_player);
        st->macro[m] = saved_macro;
        st->active = saved_active;
        st->player = saved_player;
        st->cells[m * 9 + i] = 0;
        if (v < best) best = v;
        if (v < beta) beta = v;
        if (beta <= alpha) break;
    }
    return best;
}

int32_t __stdcall sba_minimax_ultimate(const int32_t *cells,
                                       const int32_t *macro, int32_t player,
                                       int32_t active, int32_t depth,
                                       int32_t *out_best) {
    SbaState st;
    for (int32_t i = 0; i < 81; i++) st.cells[i] = cells[i];
    for (int32_t m = 0; m < SBA_N_MACROS; m++) st.macro[m] = macro[m];
    st.player = player;
    st.active = active;

    int32_t moves[81];
    int32_t n = legal_list(&st, moves);
    if (n <= 6) depth += 1;              /* dynamic depth (parity) */

    /* root ordering by eval (parity with minimax_move_ultimate) */
    double order[81];
    for (int32_t k = 0; k < n; k++) {
        SbaState t = st;
        sba_apply(&t, moves[k] / 9, moves[k] % 9);
        order[k] = eval_ultt(&t, player);
        int32_t m = moves[k] / 9, i = moves[k] % 9;
        t.cells[m * 9 + i] = 0;
        t.macro[m] = macro[m];
        t.active = active;
        t.player = player;
    }
    for (int32_t a = 0; a + 1 < n; a++) {       /* insertion sort desc */
        for (int32_t b = a + 1; b < n; b++) {
            if (order[b] > order[a]) {
                double td = order[a]; order[a] = order[b]; order[b] = td;
                int32_t tm = moves[a]; moves[a] = moves[b]; moves[b] = tm;
            }
        }
    }

    double best_score = -1e300;
    int32_t best_move = moves[0];
    for (int32_t k = 0; k < n; k++) {
        SbaState t = st;
        sba_apply(&t, moves[k] / 9, moves[k] % 9);
        double v = ab_ultt(&t, depth - 1, -1e300, 1e300, 0, player);
        if (v > best_score) {
            best_score = v;
            best_move = moves[k];
        }
    }
    if (out_best) *out_best = best_move;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Flat MCTS: independent root playouts per legal move                  */
/* ------------------------------------------------------------------ */

int32_t __stdcall sba_flat_mcts(const int32_t *cells, const int32_t *macro,
                                int32_t player, int32_t active,
                                int32_t per_move, uint64_t seed,
                                int32_t use_pattern, int32_t *out_best) {
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    SbaState st;
    for (int32_t i = 0; i < 81; i++) st.cells[i] = cells[i];
    for (int32_t m = 0; m < SBA_N_MACROS; m++) st.macro[m] = macro[m];
    st.player = player;
    st.active = active;

    int32_t moves[81];
    int32_t n = legal_list(&st, moves);
    if (n == 0) {
        if (out_best) *out_best = -1;
        return 0;
    }
    double best_wr = -1.0;
    int32_t best_move = moves[0];
    int32_t best_is_win = 0;
    for (int32_t k = 0; k < n; k++) {
        double score = 0.0;
        SbaState probe = st;
        sba_apply(&probe, moves[k] / 9, moves[k] % 9);
        /* Python flat_mcts_move breaks ties toward an immediate game win. */
        int32_t immediate_win = (sba_result(&probe) == player);
        for (int32_t r = 0; r < per_move; r++) {
            SbaState t = st;
            sba_apply(&t, moves[k] / 9, moves[k] % 9);
            int32_t res = sba_result(&t);
            if (res == 0) {
                res = use_pattern
                    ? sba_rollout_pattern(t.cells, t.macro, t.player, t.active,
                                          sba_rng_next(&rng))
                    : sba_rollout(t.cells, t.macro, t.player, t.active,
                                  sba_rng_next(&rng));
            }
            /* Python _flat_playout: win 1.0 / draw 0.5 / loss 0.0. Counting a
             * draw as a full win and dividing by 2N biased this kernel toward
             * draws and could invert the candidate ordering. */
            if (res == player) score += 1.0;
            else if (res == 3) score += 0.5;
        }
        double wr = score / (double)per_move;
        if (wr > best_wr + 1e-9 ||
            (wr >= best_wr - 1e-9 && immediate_win && !best_is_win)) {
            best_wr = wr;
            best_move = moves[k];
            best_is_win = immediate_win;
        }
    }
    if (out_best) *out_best = best_move;
    return 0;
}
