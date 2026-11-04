/*
 * sba_rollout.c — Ultimate Tic-Tac-Toe heuristic rollout kernel (D25)
 *
 * One call = one complete rollout from the given position to a terminal
 * state. The Python MCTS loop calls this once per simulation; everything
 * inside (legal-move gen, win/block threats, apply, terminal checks) runs
 * in C, avoiding the per-move Python overhead that dominates the search
 * (profiled: ~80% of MCTS time was the rollout).
 *
 * Semantics mirror ai.py's Python rollout exactly:
 *   policy  = win-if-possible (random among winning cells)
 *           > block a single opponent-winning cell
 *           > multiple threats or none: uniformly random legal move
 *   apply   = place mark; micro win claims the macro cell; a full micro
 *             board with no winner becomes a neutral draw cell (3);
 *             next active = the referenced macro if still open, else any;
 *   terminal= macro three-in-a-row, or no open macro cells left (draw).
 *
 * RNG: xorshift64* seeded from Python (random.getrandbits(64)) so runs are
 * reproducible from the Python-side random stream.
 *
 * Build (Windows cross from WSL):
 *   x86_64-w64-mingw32-gcc -O2 -shared -o sba_rollout.dll sba_rollout.c
 * See native/build.py for the automated path.
 */

#include <stdint.h>

#define N_MACROS 9

static const int32_t LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};

/* xorshift64* — returns a fresh 64-bit random number, mutates state */
static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static uint32_t rng_below(uint64_t *s, uint32_t n) {
    return n ? (uint32_t)(rng_next(s) % n) : 0;
}

/* would placing `player` on (m, i) complete a micro line in macro m? */
static int32_t wins_micro(const int32_t *cells, int32_t m, int32_t i,
                          int32_t player) {
    const int32_t *c = cells + m * 9;
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = LINES[l];
        int32_t hit = 0, other = 0;
        for (int32_t k = 0; k < 3; k++) {
            int32_t idx = ln[k];
            if (idx == i) { hit++; continue; }
            if (c[idx] == player) hit++;
            else if (c[idx] != 0) other++;
        }
        if (hit == 3 && other == 0) return 1;
    }
    return 0;
}

/* does macro cell m hold three `player` in a row? (macro: 0 open, 1 X, 2 O,
 * 3 draw — only 1/2 can win) */
static int32_t macro_line(const int32_t *macro, int32_t player) {
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = LINES[l];
        if (macro[ln[0]] == player && macro[ln[1]] == player
                && macro[ln[2]] == player) {
            return 1;
        }
    }
    return 0;
}

/*
 * cells : 81 ints, 0 empty / 1 X / 2 O
 * macro : 9 ints,  0 open / 1 X / 2 O / 3 full-draw
 * player: 1 X, 2 O — side to move at the rollout start
 * active: macro the mover is confined to, or -1 for any open macro
 * returns: 1 X wins, 2 O wins, 3 draw
 */
int32_t __stdcall sba_rollout(const int32_t *cells_in, const int32_t *macro_in,
                              int32_t player, int32_t active, uint64_t seed) {
    int32_t cells[81];
    int32_t macro[N_MACROS];
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;

    for (int32_t i = 0; i < 81; i++) cells[i] = cells_in[i];
    for (int32_t i = 0; i < N_MACROS; i++) macro[i] = macro_in[i];

    for (;;) {
        /* collect legal (m, i) pairs — macro cells with macro[m] == 0 */
        int32_t legal_m[81], legal_i[81];
        int32_t n_legal = 0;
        for (int32_t m = 0; m < N_MACROS; m++) {
            if (active >= 0 && m != active) continue;
            if (macro[m] != 0) continue;
            for (int32_t i = 0; i < 9; i++) {
                if (cells[m * 9 + i] == 0) {
                    legal_m[n_legal] = m;
                    legal_i[n_legal] = i;
                    n_legal++;
                }
            }
        }
        if (n_legal == 0) return 3; /* defensive: no open macro -> draw */

        int32_t pick = -1;

        if (player >= 1 && player <= 2) {
            /* 1) immediate micro win for the mover */
            int32_t wins_m[81], wins_i[81];
            int32_t n_wins = 0;
            for (int32_t k = 0; k < n_legal; k++) {
                if (wins_micro(cells, legal_m[k], legal_i[k], player)) {
                    wins_m[n_wins] = legal_m[k];
                    wins_i[n_wins] = legal_i[k];
                    n_wins++;
                }
            }
            if (n_wins > 0) {
                int32_t w = rng_below(&rng, (uint32_t)n_wins);
                pick = wins_m[w] * 9 + wins_i[w];
            } else {
                /* 2) block the opponent's single immediate micro win */
                int32_t opp = player == 1 ? 2 : 1;
                int32_t threats_m[81], threats_i[81];
                int32_t n_threats = 0;
                for (int32_t k = 0; k < n_legal; k++) {
                    if (wins_micro(cells, legal_m[k], legal_i[k], opp)) {
                        threats_m[n_threats] = legal_m[k];
                        threats_i[n_threats] = legal_i[k];
                        n_threats++;
                    }
                }
                if (n_threats == 1) {
                    pick = threats_m[0] * 9 + threats_i[0];
                } else {
                    /* multiple threats or none: uniform random legal */
                    int32_t r = rng_below(&rng, (uint32_t)n_legal);
                    pick = legal_m[r] * 9 + legal_i[r];
                }
            }
        } else {
            pick = legal_m[0] * 9 + legal_i[0]; /* unreachable via Python */
        }

        int32_t m = pick / 9;
        int32_t i = pick % 9;

        /* apply */
        cells[m * 9 + i] = player;
        if (wins_micro(cells, m, i, player)) {
            macro[m] = player;
        } else {
            int32_t full = 1;
            for (int32_t k = 0; k < 9; k++) {
                if (cells[m * 9 + k] == 0) { full = 0; break; }
            }
            if (full) macro[m] = 3;
        }

        /* terminal checks */
        if (macro_line(macro, player)) return player;
        int32_t any_open = 0;
        for (int32_t k = 0; k < N_MACROS; k++) {
            if (macro[k] == 0) { any_open = 1; break; }
        }
        if (!any_open) return 3;

        /* route the opponent: the referenced macro if still open, else any */
        active = macro[i] == 0 ? i : -1;
        player = player == 1 ? 2 : 1;
    }
}
