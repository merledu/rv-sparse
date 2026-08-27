/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Internal workspace layout and compiler helpers.
 */

#ifndef RVSP_COMMON_H
#define RVSP_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "rv_sparse.h"
#include "rvsp_spgemm.h"
#include "rvsp_sort.h"

#if defined(__GNUC__) || defined(__clang__)
#define RVSP_RESTRICT __restrict__
#define RVSP_LIKELY(x) __builtin_expect(!!(x), 1)
#define RVSP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define RVSP_RESTRICT
#define RVSP_LIKELY(x) (x)
#define RVSP_UNLIKELY(x) (x)
#endif

#define RVSP_ALIGN_UP(x, a) \
    (((x) + ((a) - 1)) & ~((size_t)(a) - 1))

/*
 * Column-tile width. The dense accumulator is sized by the width of B, so on
 * a wide matrix it spills out of cache and every probe becomes a memory
 * access. Tiling walks B in column strips of this many columns so the live
 * slice of the accumulator stays resident.
 *
 * 0 disables tiling and restores the single full-width pass.
 */
#ifndef RVSP_TILE_COLS
#define RVSP_TILE_COLS 0
#endif

#if RVSP_TILE_COLS < 0
#error "RVSP_TILE_COLS must be >= 0 (0 disables tiling)"
#endif

/*
 * Scratch shared across a call. acc is the dense accumulator, mark and
 * touched track which columns are live during the symbolic pass, and scratch
 * backs the column sort. Each region is 64-byte aligned in one buffer.
 */
typedef struct
{
    float *acc;
    uint8_t *mark;
    int32_t *touched;
    int32_t *scratch;
} rvsp_ws_t;

static inline size_t
rvsp_ws_bytes(int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    return RVSP_ALIGN_UP(n * sizeof(float), 64) +
           RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_ws_bind(
    rvsp_ws_t *ws,
    void *buffer,
    int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = (float *)p;
    p += RVSP_ALIGN_UP(n * sizeof(float), 64);

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->touched = (int32_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(int32_t), 64);

    ws->scratch = (int32_t *)p;
}

/* Symbolic count needs only mark and touched. */
static inline size_t
rvsp_count_ws_bytes(int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    return RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_count_ws_bind(
    rvsp_ws_t *ws,
    void *buffer,
    int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = NULL;

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->touched = (int32_t *)p;
    ws->scratch = NULL;
}

/* Numeric compute needs acc, mark, and scratch but not touched. */
static inline size_t
rvsp_compute_ws_bytes(int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    return RVSP_ALIGN_UP(n * sizeof(float), 64) +
           RVSP_ALIGN_UP(n * sizeof(uint8_t), 64) +
           RVSP_ALIGN_UP(n * sizeof(int32_t), 64);
}

static inline void
rvsp_compute_ws_bind(
    rvsp_ws_t *ws,
    void *buffer,
    int32_t b_cols)
{
    const size_t n = (size_t)(b_cols > 0 ? b_cols : 1);

    unsigned char *p = (unsigned char *)buffer;

    ws->acc = (float *)p;
    p += RVSP_ALIGN_UP(n * sizeof(float), 64);

    ws->mark = (uint8_t *)p;
    p += RVSP_ALIGN_UP(n * sizeof(uint8_t), 64);

    ws->scratch = (int32_t *)p;
    ws->touched = NULL;
}

#endif