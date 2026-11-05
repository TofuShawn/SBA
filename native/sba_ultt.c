/*
 * sba_ultt.c — shared rules core for the SBA native engine (D27).
 */

#include "sba_ultt.h"

const int32_t SBA_LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};

uint64_t sba_rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

uint32_t sba_rng_below(uint64_t *s, uint32_t n) {
    return n ? (uint32_t)(sba_rng_next(s) % n) : 0;
}

int32_t sba_wins_micro(const int32_t *cells, int32_t m, int32_t i,
                       int32_t player) {
    const int32_t *c = cells + m * 9;
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = SBA_LINES[l];
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

int32_t sba_macro_line(const int32_t *macro, int32_t player) {
    for (int32_t l = 0; l < 8; l++) {
        const int32_t *ln = SBA_LINES[l];
        if (macro[ln[0]] == player && macro[ln[1]] == player
                && macro[ln[2]] == player) {
            return 1;
        }
    }
    return 0;
}

int32_t sba_apply(SbaState *st, int32_t m, int32_t i) {
    st->cells[m * 9 + i] = st->player;
    if (sba_wins_micro(st->cells, m, i, st->player)) {
        st->macro[m] = st->player;
    } else {
        int32_t full = 1;
        for (int32_t k = 0; k < 9; k++) {
            if (st->cells[m * 9 + k] == 0) { full = 0; break; }
        }
        if (full) st->macro[m] = 3;
    }
    int32_t done = sba_macro_line(st->macro, st->player) ? st->player : 0;
    if (!done) {
        int32_t any_open = 0;
        for (int32_t k = 0; k < SBA_N_MACROS; k++) {
            if (st->macro[k] == 0) { any_open = 1; break; }
        }
        if (!any_open) done = 3;
    }
    st->active = st->macro[i] == 0 ? i : -1;
    st->player = st->player == 1 ? 2 : 1;
    return done;
}

int32_t sba_result(const SbaState *st) {
    if (sba_macro_line(st->macro, 1)) return 1;
    if (sba_macro_line(st->macro, 2)) return 2;
    for (int32_t k = 0; k < SBA_N_MACROS; k++) {
        if (st->macro[k] == 0) return 0;
    }
    return 3;
}
