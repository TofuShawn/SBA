/*
 * sba_mcts.c — full plain MCTS in C (D27/D37).
 *
 * D37 changes:
 *   - the node arena is owned by the DLL (malloc/calloc), so the caller no
 *     longer passes a byte buffer that gets cast to MctsNode*; a tree is
 *     handed back as an opaque SbaTreeHandle and released with sba_tree_free.
 *   - SbaSearchOpts carries every switch that used to be Python-only:
 *     uct_c, dynamic_uct, rollout_heuristic, symmetry, use_pattern
 *     (micro_tablebase belongs to the minimax evaluator, see sba_minimax.c).
 *   - root symmetry dedup compares the CANONICAL RESULTING POSITION (cells +
 *     macro + active routing + side to move) under the 8 D4 transforms, not
 *     the move's cell orbit — the orbit key merged non-equivalent moves.
 *
 * Node semantics still mirror ai.py's MCTSNode (wins from the mover's view,
 * draws counted separately) and the best-move tie-break checks a GAME win.
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
    int32_t draws;         /* playouts through this node that ended in a draw */
    uint64_t untried[2];   /* bitmask of not-yet-expanded moves (81 bits) */
    int32_t cells[81];
    int32_t macro[SBA_N_MACROS];
    int32_t player;
    int32_t active;
} MctsNode;

/* A C-owned search tree. malloc() gives suitably aligned storage, so the
 * uint8_t* -> MctsNode* cast (and its alignment assumption) is gone. */
typedef struct {
    MctsNode *arena;
    int32_t cap;
    int32_t n;
} SbaTree;

/* Same 8 D4 permutations as ai.py's _D4 (verified against it). */
static const int32_t SBA_D4[8][9] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8},
    {2, 5, 8, 1, 4, 7, 0, 3, 6},
    {8, 7, 6, 5, 4, 3, 2, 1, 0},
    {6, 3, 0, 7, 4, 1, 8, 5, 2},
    {6, 7, 8, 3, 4, 5, 0, 1, 2},
    {8, 5, 2, 7, 4, 1, 6, 3, 0},
    {2, 1, 0, 5, 4, 3, 8, 7, 6},
    {0, 3, 6, 1, 4, 7, 2, 5, 8},
};

static const SbaSearchOpts SBA_DEFAULT_OPTS = {
    1.4,  /* uct_c */
    1,    /* dynamic_uct */
    1,    /* rollout_heuristic */
    1,    /* micro_tablebase */
    1,    /* symmetry */
    0,    /* use_pattern */
};

static double uct_scale(double c, int32_t dynamic, double root_visits,
                        int32_t iterations) {
    /* ai.py _uct_scale: c * (0.6 + 0.4 * (1 - root_visits / iterations)) */
    if (!dynamic) return c;
    double limit = iterations > 0 ? (double)iterations : 1.0;
    return c * (0.6 + 0.4 * (1.0 - root_visits / limit));
}

/* Two resulting positions are equivalent iff some D4 transform maps one to
 * the other, including macro statuses, the forced next macro and the side to
 * move. Exact comparison (no hashing), so no collision risk. */
static int32_t d4_equivalent(const MctsNode *a, const MctsNode *b) {
    for (int32_t t = 0; t < 8; t++) {
        const int32_t *p = SBA_D4[t];
        int32_t ok = 1;
        for (int32_t m = 0; m < SBA_N_MACROS && ok; m++) {
            for (int32_t i = 0; i < 9 && ok; i++) {
                if (a->cells[m * 9 + i] != b->cells[p[m] * 9 + p[i]]) ok = 0;
            }
        }
        for (int32_t m = 0; m < SBA_N_MACROS && ok; m++) {
            if (a->macro[m] != b->macro[p[m]]) ok = 0;
        }
        if (ok) {
            int32_t active_b = (b->active >= 0) ? p[b->active] : -1;
            if (a->active != active_b || a->player != b->player) ok = 0;
        }
        if (ok) return 1;
    }
    return 0;
}

static SbaTree *tree_alloc(int32_t cap) {
    SbaTree *t = (SbaTree *)calloc(1, sizeof(SbaTree));
    if (!t) return NULL;
    t->arena = (MctsNode *)calloc((size_t)cap, sizeof(MctsNode));
    if (!t->arena) { free(t); return NULL; }
    t->cap = cap;
    t->n = 0;
    return t;
}

int32_t __stdcall sba_tree_free(SbaTreeHandle handle) {
    SbaTree *t = (SbaTree *)handle;
    if (!t) return 0;
    free(t->arena);
    free(t);
    return 0;
}

/* Recursively copy the subtree rooted at src_idx into dst (remapping
 * parent/child/next indices). Returns the new index, or -1 on overflow. */
static int32_t copy_subtree(SbaTree *dst, const SbaTree *src, int32_t src_idx,
                            int32_t dst_parent) {
    if (dst->n >= dst->cap) return -1;
    int32_t d = dst->n++;
    MctsNode *n = &dst->arena[d];
    const MctsNode *s = &src->arena[src_idx];
    *n = *s;
    n->parent = dst_parent;
    n->child = n->next = -1;
    for (int32_t ch = s->child; ch != -1; ch = src->arena[ch].next) {
        int32_t c = copy_subtree(dst, src, ch, d);
        if (c < 0) return -1;
        dst->arena[c].next = n->child;
        n->child = c;
    }
    return d;
}

static void legal_mask_into(uint64_t out[2], const SbaState *st) {
    out[0] = out[1] = 0;
    if (sba_result(st) != 0) return;
    for (int32_t m = 0; m < SBA_N_MACROS; m++) {
        if (st->active >= 0 && m != st->active) continue;
        if (st->macro[m] != 0) continue;
        for (int32_t i = 0; i < 9; i++) {
            if (st->cells[m * 9 + i] == 0) {
                int32_t b = m * 9 + i;
                if (b < 64) out[0] |= (uint64_t)1 << b;
                else out[1] |= (uint64_t)1 << (b - 64);
            }
        }
    }
}

static void state_from_node(SbaState *st, const MctsNode *n) {
    for (int32_t i = 0; i < 81; i++) st->cells[i] = n->cells[i];
    for (int32_t m = 0; m < SBA_N_MACROS; m++) st->macro[m] = n->macro[m];
    st->player = n->player;
    st->active = n->active;
}

static void node_from_state(MctsNode *n, const SbaState *st) {
    for (int32_t i = 0; i < 81; i++) n->cells[i] = st->cells[i];
    for (int32_t m = 0; m < SBA_N_MACROS; m++) n->macro[m] = st->macro[m];
    n->player = st->player;
    n->active = st->active;
}

int32_t __stdcall sba_mcts_search(const int32_t *cells, const int32_t *macro,
                                  int32_t player, int32_t active,
                                  int32_t iterations, uint64_t seed,
                                  const SbaSearchOpts *opts_in,
                                  int32_t *out_best, int32_t *out_count,
                                  int32_t *out_moves, double *out_visits,
                                  double *out_wins, int32_t *out_mover,
                                  int32_t *out_draws,
                                  SbaTreeHandle *out_tree,
                                  const SbaTreeHandle prev_tree) {
    const SbaSearchOpts *opts = opts_in ? opts_in : &SBA_DEFAULT_OPTS;
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    if (out_tree) *out_tree = NULL;
    if (iterations <= 0) return -2;

    /* 2x headroom: a reused subtree plus this search's new nodes. */
    SbaTree *tree = tree_alloc(iterations * 2 + 64);
    if (!tree) return -2;
    MctsNode *arena = tree->arena;
    int32_t max_nodes = tree->cap;
    MctsNode *root = NULL;

    /* --- tree reuse (D30) --- */
    if (prev_tree != NULL) {
        const SbaTree *prev = (const SbaTree *)prev_tree;
        const MctsNode *par = prev->arena;
        int32_t cur = 0, matched = 0;
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
            int32_t new_root = copy_subtree(tree, prev, cur, -1);
            if (new_root >= 0 && tree->n + iterations + 8 <= max_nodes) {
                root = &arena[new_root];
                root->parent = -1;
            } else {
                tree->n = 0;   /* overflow: start from a fresh root */
            }
        }
    }

    if (root == NULL) {
        root = &arena[tree->n++];
        root->parent = root->child = root->next = -1;
        root->move = -1;
        root->mover = 0;   /* nobody moved into the root: Python parity (None) */
        root->visits = 0;
        root->wins = 0.0;
        root->draws = 0;
        root->untried[0] = root->untried[1] = 0;
        root->player = player;
        root->active = active;
        for (int32_t i = 0; i < 81; i++) root->cells[i] = cells[i];
        for (int32_t m = 0; m < SBA_N_MACROS; m++) root->macro[m] = macro[m];
        SbaState st;
        state_from_node(&st, root);
        uint64_t mask[2];
        legal_mask_into(mask, &st);
        root->untried[0] = mask[0];
        root->untried[1] = mask[1];
    }

    for (int32_t it = 0; it < iterations; it++) {
        /* --- selection --- */
        MctsNode *node = root;
        while (1) {
            if ((node->untried[0] | node->untried[1]) != 0) break;
            if (node->child == -1) break;
            double log_n = log((double)(node->visits > 0 ? node->visits : 1));
            double scale = uct_scale(opts->uct_c, opts->dynamic_uct,
                                     (double)root->visits, iterations);
            int32_t best = -1;
            double best_val = -1e300;
            for (int32_t ch = node->child; ch != -1; ch = arena[ch].next) {
                double val;
                if (arena[ch].visits == 0) val = 1e300;
                else val = arena[ch].wins / arena[ch].visits
                           + scale * sqrt(log_n / arena[ch].visits);
                if (val > best_val) { best_val = val; best = ch; }
            }
            if (best < 0) break;
            node = &arena[best];
        }

        /* --- expansion: pop random untried moves until one is accepted --- */
        if ((node->untried[0] | node->untried[1]) != 0 && tree->n < max_nodes) {
            int32_t accepted = 0;
            while (!accepted && (node->untried[0] | node->untried[1]) != 0
                   && tree->n < max_nodes) {
                uint64_t un0 = node->untried[0], un1 = node->untried[1];
                int32_t total = 0;
                {
                    uint64_t t0 = un0, t1 = un1;
                    while (t0) { t0 &= t0 - 1; total++; }
                    while (t1) { t1 &= t1 - 1; total++; }
                }
                int32_t pick = (int32_t)sba_rng_below(&rng, (uint32_t)total);
                int32_t idx = -1, seen = 0;
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
                if (idx < 0) break;

                MctsNode *child = &arena[tree->n++];
                child->parent = (int32_t)(node - arena);
                child->child = child->next = -1;
                child->move = idx;
                child->mover = node->player;
                child->visits = 0;
                child->wins = 0.0;
                child->draws = 0;
                child->untried[0] = child->untried[1] = 0;
                for (int32_t i = 0; i < 81; i++) child->cells[i] = node->cells[i];
                for (int32_t m = 0; m < SBA_N_MACROS; m++)
                    child->macro[m] = node->macro[m];
                child->player = node->player;
                child->active = node->active;
                SbaState st;
                state_from_node(&st, child);
                sba_apply(&st, idx / 9, idx % 9);
                node_from_state(child, &st);
                uint64_t cmask[2];
                legal_mask_into(cmask, &st);
                child->untried[0] = cmask[0];
                child->untried[1] = cmask[1];
                child->next = node->child;
                node->child = (int32_t)(child - arena);

                if (opts->symmetry && node == root) {
                    int32_t duplicate = 0;
                    for (int32_t ch = child->next; ch != -1; ch = arena[ch].next) {
                        if (d4_equivalent(&arena[ch], child)) {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (duplicate) {
                        node->child = child->next;   /* unlink */
                        tree->n--;                    /* roll back the slot */
                        continue;                     /* try another move */
                    }
                }
                node = child;
                accepted = 1;
            }
        }

        /* --- rollout --- */
        int32_t result;
        {
            SbaState st;
            state_from_node(&st, node);
            result = sba_result(&st);
        }
        if (result == 0) {
            uint64_t rseed = sba_rng_next(&rng);
            if (!opts->rollout_heuristic) {
                result = sba_rollout_random(node->cells, node->macro,
                                            node->player, node->active, rseed);
            } else if (opts->use_pattern) {
                result = sba_rollout_pattern(node->cells, node->macro,
                                             node->player, node->active, rseed);
            } else {
                result = sba_rollout(node->cells, node->macro,
                                     node->player, node->active, rseed);
            }
        }

        /* --- backprop --- */
        MctsNode *bn = node;
        while (bn != NULL) {
            bn->visits++;
            if (result == 3) { bn->wins += 0.5; bn->draws += 1; }
            else if (result == bn->mover) bn->wins += 1.0;
            bn = bn->parent >= 0 ? &arena[bn->parent] : NULL;
        }
    }

    /* --- best move (parity with _best_mcts_move) --- */
    int32_t min_visits = iterations / 100 > 5 ? iterations / 100 : 5;
    double best_wr = -1.0;
    int32_t best_idx = -1;
    for (int32_t ch = root->child; ch != -1; ch = arena[ch].next) {
        if (arena[ch].visits < min_visits) continue;
        double wr = arena[ch].wins / arena[ch].visits;
        if (wr > best_wr) { best_wr = wr; best_idx = arena[ch].move; }
    }
    for (int32_t ch = root->child; ch != -1; ch = arena[ch].next) {
        if (arena[ch].visits < min_visits) continue;
        int32_t m = arena[ch].move / 9, i = arena[ch].move % 9;
        /* a GAME win (macro line completed), matching Python's g.result():
         * a bare micro win that claims no line must not count */
        int32_t wins_game = 0;
        if (sba_wins_micro(root->cells, m, i, root->player)) {
            int32_t saved = root->macro[m];
            root->macro[m] = root->player;
            wins_game = sba_macro_line(root->macro, root->player);
            root->macro[m] = saved;
        }
        if (wins_game && arena[ch].wins / arena[ch].visits == best_wr) {
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
        if (out_draws) out_draws[cnt] = arena[ch].draws;
        cnt++;
    }
    if (out_count) *out_count = cnt;
    if (out_best) *out_best = best_idx;
    if (out_tree) *out_tree = (SbaTreeHandle)tree;
    return 0;
}
