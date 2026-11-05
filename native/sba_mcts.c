/*
 * sba_mcts.c — full plain MCTS in C (D27): arena tree, UCT selection with
 * the Python dynamic-uct annealing, expansion, in-process rollout (plain or
 * pattern), backprop, and root stats export so analyze_position /
 * position_win_rates can reuse one search exactly like the Python path.
 *
 * Node semantics mirror ai.py's MCTSNode:
 *   - wins are counted from the mover's perspective (1 win / 0.5 draw)
 *   - expansion pops a random untried move
 *   - best move = highest win-rate among sufficiently visited children,
 *     preferring an immediate game-winning move (parity with
 *     _best_mcts_move).
 *
 * Tree reuse is NOT carried over (the Python tree is not transferable);
 * the C search is fast enough that this nets out strongly positive.
 */

#include "sba_ultt.h"
#include <stdlib.h>
#include <math.h>

typedef struct {
    int32_t parent, child, next;
    int32_t move;
    int32_t mover;
    int32_t visits;
    double wins;
    uint64_t untried[2];   /* bitmask of not-yet-expanded moves (81 bits) */
    int32_t cells[81];
    int32_t macro[SBA_N_MACROS];
    int32_t player;
    int32_t active;
} MctsNode;

static double uct_scale(int32_t c_times_100, double root_visits,
                        int32_t iterations) {
    double c = c_times_100 / 100.0;
    /* ai.py _uct_scale: c * (0.6 + 0.4 * (1 - root_visits / iterations)) */
    return c * (0.6 + 0.4 * (1.0 - root_visits / (iterations > 0 ? iterations : 1)));
}

/* Recursively copy the subtree rooted at src into dst arena (remapping
 * parent/child/next indices). Returns the new index, or -1 when the arena
 * capacity is exceeded (caller then starts from a fresh root). */
static int32_t copy_subtree(MctsNode *dst, int32_t *dn, int32_t cap,
                            const MctsNode *src_arena, int32_t src,
                            int32_t dst_parent) {
    if (*dn >= cap) return -1;
    int32_t d = (*dn)++;
    MctsNode *n = &dst[d];
    const MctsNode *s = &src_arena[src];
    n->parent = dst_parent;
    n->child = n->next = -1;
    n->move = s->move;
    n->mover = s->mover;
    n->visits = s->visits;
    n->wins = s->wins;
    n->untried[0] = s->untried[0];
    n->untried[1] = s->untried[1];
    for (int32_t i = 0; i < 81; i++) n->cells[i] = s->cells[i];
    for (int32_t m = 0; m < SBA_N_MACROS; m++) n->macro[m] = s->macro[m];
    n->player = s->player;
    n->active = s->active;
    for (int32_t ch = s->child; ch != -1; ch = src_arena[ch].next) {
        int32_t c = copy_subtree(dst, dn, cap, src_arena, ch, d);
        if (c < 0) return -1;
        dst[c].next = n->child;
        n->child = c;
    }
    return d;
}

int32_t __stdcall sba_mcts_search(const int32_t *cells, const int32_t *macro,
                                  int32_t player, int32_t active,
                                  int32_t iterations, uint64_t seed,
                                  int32_t use_pattern, int32_t *out_best,
                                  int32_t *out_count, int32_t *out_moves,
                                  double *out_visits, double *out_wins,
                                  int32_t *out_mover, uint8_t *buf,
                                  int32_t buf_bytes,
                                  const uint8_t *prev_buf,
                                  int32_t prev_bytes) {
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    /* No CRT in this DLL (mingw cross builds lack a working one here): the
     * caller supplies the node arena. Capacity check -> -2 when too small. */
    int32_t max_nodes = (int32_t)(buf_bytes / sizeof(MctsNode));
    if (max_nodes < iterations + 64) return -2;
    MctsNode *arena = (MctsNode *)buf;
    int32_t n_nodes = 0;
    MctsNode *root = NULL;

    /* --- tree reuse (D30): walk the previous tree down to the current
     * position (one differing cell per ply), then copy the matched subtree
     * in as the new root. Falls back to a fresh root on any mismatch, and
     * bails (-> -3) when the subtree does not fit the arena. --- */
    if (prev_buf != NULL && prev_bytes >= (int32_t)sizeof(MctsNode)) {
        const MctsNode *par = (const MctsNode *)prev_buf;
        int32_t prev_cap = prev_bytes / (int32_t)sizeof(MctsNode);
        int32_t cur = 0;
        int32_t matched = 0;
        for (int32_t guard = 0; guard < 128; guard++) {
            int32_t diffs = 0, diff_idx = -1;
            for (int32_t i = 0; i < 81; i++) {
                if (par[cur].cells[i] != cells[i]) { diffs++; diff_idx = i; }
            }
            if (diffs == 0) { matched = 1; break; }
            if (diffs != 1) break;
            int32_t c = -1;
            for (int32_t ch = par[cur].child; ch != -1; ch = par[ch].next) {
                if (par[ch].move == diff_idx) { c = ch; break; }
            }
            if (c < 0) break;
            cur = c;
        }
        if (matched && par[cur].player == player) {
            int32_t dn = 0;
            int32_t new_root = copy_subtree(arena, &dn, max_nodes, par, cur, -1);
            if (new_root >= 0 && dn + iterations + 8 <= max_nodes) {
                n_nodes = dn;
                root = &arena[new_root];
                root->parent = -1;
            }
        }
    }

    if (root == NULL) {
        root = &arena[n_nodes++];
        root->parent = root->child = root->next = -1;
        root->move = -1;
        root->mover = 0;   /* nobody moved into the root: Python parity (None) */
        root->visits = 0;
        root->wins = 0.0;
        root->untried[0] = root->untried[1] = 0;
        root->player = player;
        root->active = active;
        for (int32_t i = 0; i < 81; i++) root->cells[i] = cells[i];
        for (int32_t m = 0; m < SBA_N_MACROS; m++) root->macro[m] = macro[m];
        /* root untried mask = legal moves */
        {
            SbaState st;
            for (int32_t i = 0; i < 81; i++) st.cells[i] = cells[i];
            for (int32_t m = 0; m < SBA_N_MACROS; m++) st.macro[m] = macro[m];
            st.player = player;
            st.active = active;
            uint64_t mask[2] = {0, 0};
            for (int32_t m = 0; m < SBA_N_MACROS; m++) {
                if (active >= 0 && m != active) continue;
                if (macro[m] != 0) continue;
                for (int32_t i = 0; i < 9; i++) {
                    if (cells[m * 9 + i] == 0) {
                        int32_t idx = m * 9 + i;
                        if (idx < 64) mask[0] |= (uint64_t)1 << idx;
                        else mask[1] |= (uint64_t)1 << (idx - 64);
                    }
                }
            }
            root->untried[0] = mask[0];
            root->untried[1] = mask[1];
        }
    }

    for (int32_t it = 0; it < iterations; it++) {
        /* --- selection: walk down fully-expanded nodes by UCT --- */
        MctsNode *node = root;
        while (1) {
            if ((node->untried[0] | node->untried[1]) != 0) break;
            if (node->child == -1) break;
            double log_n = log((double)(node->visits > 0 ? node->visits : 1));
            double scale = uct_scale(140, (double)root->visits, iterations);
            int32_t best = -1;
            double best_val = -1e300;
            for (int32_t ch = node->child; ch != -1; ch = arena[ch].next) {
                double val;
                if (arena[ch].visits == 0) val = 1e300;
                else val = arena[ch].wins / arena[ch].visits
                           + scale * sqrt(log_n / arena[ch].visits);
                if (val > best_val) { best_val = val; best = ch; }
            }
            node = &arena[best];
        }

        /* --- expansion: pop a random untried move --- */
        if ((node->untried[0] | node->untried[1]) != 0 && n_nodes < max_nodes) {
            uint64_t un0 = node->untried[0], un1 = node->untried[1];
            int32_t total = 0;
            {
                uint64_t t0 = un0, t1 = un1;
                while (t0) { t0 &= t0 - 1; total++; }
                while (t1) { t1 &= t1 - 1; total++; }
            }
            int32_t pick = (int32_t)sba_rng_below(&rng, (uint32_t)total);
            int32_t idx = -1;
            int32_t seen = 0;
            for (int32_t b = 0; b < 81; b++) {
                uint64_t *w = b < 64 ? &un0 : &un1;
                int32_t sh = b < 64 ? b : b - 64;
                if ((*w >> sh) & 1) {
                    if (seen == pick) {
                        idx = b;
                        *w &= ~((uint64_t)1 << sh);
                        break;
                    }
                    seen++;
                }
            }
            node->untried[0] = un0;
            node->untried[1] = un1;

            MctsNode *child = &arena[n_nodes++];
            child->parent = (int32_t)(node - arena);
            child->child = child->next = -1;
            child->move = idx;
            child->mover = node->player;
            child->visits = 0;
            child->wins = 0.0;
            child->untried[0] = child->untried[1] = 0;
            for (int32_t i = 0; i < 81; i++) child->cells[i] = node->cells[i];
            for (int32_t m = 0; m < SBA_N_MACROS; m++)
                child->macro[m] = node->macro[m];
            child->player = node->player;
            child->active = node->active;
            SbaState st;
            for (int32_t i = 0; i < 81; i++) st.cells[i] = child->cells[i];
            for (int32_t m = 0; m < SBA_N_MACROS; m++)
                st.macro[m] = child->macro[m];
            st.player = child->player;
            st.active = child->active;
            sba_apply(&st, idx / 9, idx % 9);
            for (int32_t i = 0; i < 81; i++) child->cells[i] = st.cells[i];
            for (int32_t m = 0; m < SBA_N_MACROS; m++)
                child->macro[m] = st.macro[m];
            child->player = st.player;
            child->active = st.active;
            uint64_t cmask[2] = {0, 0};
            if (sba_result(&st) == 0) {
                for (int32_t m = 0; m < SBA_N_MACROS; m++) {
                    if (st.active >= 0 && m != st.active) continue;
                    if (st.macro[m] != 0) continue;
                    for (int32_t i = 0; i < 9; i++) {
                        if (st.cells[m * 9 + i] == 0) {
                            int32_t b = m * 9 + i;
                            if (b < 64) cmask[0] |= (uint64_t)1 << b;
                            else cmask[1] |= (uint64_t)1 << (b - 64);
                        }
                    }
                }
            }
            child->untried[0] = cmask[0];
            child->untried[1] = cmask[1];
            child->next = node->child;
            node->child = (int32_t)(child - arena);
            node = child;
        }

        /* --- rollout: one terminal playout from the node state --- */
        int32_t result;
        {
            SbaState st;
            for (int32_t i = 0; i < 81; i++) st.cells[i] = node->cells[i];
            for (int32_t m = 0; m < SBA_N_MACROS; m++) st.macro[m] = node->macro[m];
            st.player = node->player;
            st.active = node->active;
            result = sba_result(&st);
        }
        if (result == 0) {
            result = use_pattern
                ? sba_rollout_pattern(node->cells, node->macro, node->player,
                                      node->active, sba_rng_next(&rng))
                : sba_rollout(node->cells, node->macro, node->player,
                              node->active, sba_rng_next(&rng));
        }

        /* --- backprop (wins from the mover's perspective) --- */
        MctsNode *bn = node;
        while (bn != NULL) {
            bn->visits++;
            if (result == bn->mover) bn->wins += 1.0;
            else if (result == 3) bn->wins += 0.5;
            bn = bn->parent >= 0 ? &arena[bn->parent] : NULL;
        }
    }

    /* --- pick the best move (parity with _best_mcts_move) --- */
    int32_t min_visits = iterations / 100 > 5 ? iterations / 100 : 5;
    double best_wr = -1.0;
    int32_t best_idx = -1;
    for (int32_t ch = root->child; ch != -1; ch = arena[ch].next) {
        if (arena[ch].visits < min_visits) continue;
        double wr = arena[ch].wins / arena[ch].visits;
        if (wr > best_wr) { best_wr = wr; best_idx = arena[ch].move; }
    }
    /* prefer an immediate game-winning move among the visited children */
    for (int32_t ch = root->child; ch != -1; ch = arena[ch].next) {
        if (arena[ch].visits < min_visits) continue;
        int32_t m = arena[ch].move / 9, i = arena[ch].move % 9;
        if (sba_wins_micro(root->cells, m, i, root->player)
                && arena[ch].wins / arena[ch].visits == best_wr) {
            best_idx = arena[ch].move;
            break;
        }
    }
    if (best_idx < 0 && root->child != -1) best_idx = arena[root->child].move;

    /* --- export root stats --- */
    int32_t cnt = 0;
    for (int32_t ch = root->child; ch != -1; ch = arena[ch].next) {
        if (out_moves) out_moves[cnt] = arena[ch].move;
        if (out_visits) out_visits[cnt] = (double)arena[ch].visits;
        if (out_wins) out_wins[cnt] = arena[ch].wins;
        if (out_mover) out_mover[cnt] = arena[ch].mover;
        cnt++;
    }
    if (out_count) *out_count = cnt;
    if (out_best) *out_best = best_idx;

    return 0;
}
