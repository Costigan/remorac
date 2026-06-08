/* Remora C runtime library — sorting, filtering, replication for CPU backend.

   The LLVM ABI flattens memref descriptors into individual parameters:
   (allocated_ptr, aligned_ptr, offset, sizes[0], strides[0])
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Memref helpers ─────────────────────────────────────────────────────── */

static inline void* _mr_data(void* aligned, int64_t offset) {
    return (char*)aligned + offset;
}

/* ── Comparison helpers for qsort ──────────────────────────────────────── */

static int _cmp_i32_asc(const void* a, const void* b) {
    int32_t va = *(const int32_t*)a;
    int32_t vb = *(const int32_t*)b;
    return (va > vb) - (va < vb);
}

static int _cmp_f32_asc(const void* a, const void* b) {
    float va = *(const float*)a;
    float vb = *(const float*)b;
    return (va > vb) - (va < vb);
}

/* ── Sort (in-place) ────────────────────────────────────────────────────── */
/* LLVM ABI: (allocated, aligned, offset, size, stride) */

void remora_sort_i32(
    int32_t* allocated, int32_t* aligned, int64_t offset, int64_t size, int64_t stride
) {
    (void)allocated; (void)stride;
    qsort(_mr_data(aligned, offset), (size_t)size, sizeof(int32_t), _cmp_i32_asc);
}

void remora_sort_f32(
    float* allocated, float* aligned, int64_t offset, int64_t size, int64_t stride
) {
    (void)allocated; (void)stride;
    qsort(_mr_data(aligned, offset), (size_t)size, sizeof(float), _cmp_f32_asc);
}

/* ── Grade (argsort) ────────────────────────────────────────────────────── */

typedef struct {
    void*  base;
    int    index;
} _grade_pair_t;

static int _cmp_grade_i32(const void* a, const void* b) {
    const _grade_pair_t* ga = (const _grade_pair_t*)a;
    const _grade_pair_t* gb = (const _grade_pair_t*)b;
    int32_t va = *(const int32_t*)ga->base;
    int32_t vb = *(const int32_t*)gb->base;
    if (va != vb) return (va > vb) - (va < vb);
    return ga->index - gb->index;
}

static int _cmp_grade_f32(const void* a, const void* b) {
    const _grade_pair_t* ga = (const _grade_pair_t*)a;
    const _grade_pair_t* gb = (const _grade_pair_t*)b;
    float va = *(const float*)ga->base;
    float vb = *(const float*)gb->base;
    if (va != vb) return (va > vb) - (va < vb);
    return ga->index - gb->index;
}

void remora_grade_i32(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)dst_alloc; (void)dst_n; (void)dst_str;
    int32_t* src_data = (int32_t*)_mr_data(src_align, src_off);
    int32_t* dst_data = (int32_t*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    _grade_pair_t* pairs = (_grade_pair_t*)malloc((size_t)n * sizeof(_grade_pair_t));
    for (int64_t i = 0; i < n; i++) {
        pairs[i].base = &src_data[i];
        pairs[i].index = (int)i;
    }
    qsort(pairs, (size_t)n, sizeof(_grade_pair_t), _cmp_grade_i32);
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = (int32_t)pairs[i].index;
    }
    free(pairs);
}

void remora_grade_f32(
    float* src_alloc, float* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)dst_alloc; (void)dst_n; (void)dst_str;
    float*   src_data = (float*)_mr_data(src_align, src_off);
    int32_t* dst_data = (int32_t*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    _grade_pair_t* pairs = (_grade_pair_t*)malloc((size_t)n * sizeof(_grade_pair_t));
    for (int64_t i = 0; i < n; i++) {
        pairs[i].base = &src_data[i];
        pairs[i].index = (int)i;
    }
    qsort(pairs, (size_t)n, sizeof(_grade_pair_t), _cmp_grade_f32);
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = (int32_t)pairs[i].index;
    }
    free(pairs);
}

/* ── Filter (dynamic output size, returns actual count) ────────────────── */

int64_t remora_filter_i32(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* mask_alloc, int32_t* mask_align, int64_t mask_off, int64_t mask_n, int64_t mask_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)mask_alloc; (void)mask_n; (void)mask_str;
    (void)dst_alloc; (void)dst_n; (void)dst_str;
    int32_t* src_data  = (int32_t*)_mr_data(src_align, src_off);
    int32_t* mask_data = (int32_t*)_mr_data(mask_align, mask_off);
    int32_t* dst_data  = (int32_t*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    int64_t out_n = 0;
    for (int64_t i = 0; i < n; i++) {
        if (mask_data[i]) {
            dst_data[out_n++] = src_data[i];
        }
    }
    return out_n;
}

int64_t remora_filter_f32(
    float* src_alloc, float* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* mask_alloc, int32_t* mask_align, int64_t mask_off, int64_t mask_n, int64_t mask_str,
    float* dst_alloc, float* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)mask_alloc; (void)mask_n; (void)mask_str;
    (void)dst_alloc; (void)dst_n; (void)dst_str;
    float*   src_data  = (float*)_mr_data(src_align, src_off);
    int32_t* mask_data = (int32_t*)_mr_data(mask_align, mask_off);
    float*   dst_data  = (float*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    int64_t out_n = 0;
    for (int64_t i = 0; i < n; i++) {
        if (mask_data[i]) {
            dst_data[out_n++] = src_data[i];
        }
    }
    return out_n;
}

/* ── Replicate count helper (compute output size without filling) ───────── */

int64_t remora_replicate_i32_count(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* cnt_alloc, int32_t* cnt_align, int64_t cnt_off, int64_t cnt_n, int64_t cnt_str
) {
    (void)src_alloc; (void)src_align; (void)src_off; (void)src_str;
    (void)cnt_alloc; (void)cnt_align; (void)cnt_off; (void)cnt_str;
    int32_t* cnt_data = (int32_t*)_mr_data(cnt_align, cnt_off);
    int64_t n = src_n;
    int64_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        total += cnt_data[i];
    }
    return total;
}

int64_t remora_replicate_f32_count(
    float* src_alloc, float* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* cnt_alloc, int32_t* cnt_align, int64_t cnt_off, int64_t cnt_n, int64_t cnt_str
) {
    (void)src_alloc; (void)src_align; (void)src_off; (void)src_str;
    (void)cnt_alloc; (void)cnt_align; (void)cnt_off; (void)cnt_str;
    int32_t* cnt_data = (int32_t*)_mr_data(cnt_align, cnt_off);
    int64_t n = src_n;
    int64_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        total += cnt_data[i];
    }
    return total;
}

/* ── Replicate (dynamic output size, returns actual count) ──────────────── */

int64_t remora_replicate_i32(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* cnt_alloc, int32_t* cnt_align, int64_t cnt_off, int64_t cnt_n, int64_t cnt_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)cnt_alloc; (void)cnt_n; (void)cnt_str;
    (void)dst_alloc; (void)dst_n; (void)dst_str;
    int32_t* src_data  = (int32_t*)_mr_data(src_align, src_off);
    int32_t* cnt_data  = (int32_t*)_mr_data(cnt_align, cnt_off);
    int32_t* dst_data  = (int32_t*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    int64_t out_n = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t count = cnt_data[i];
        for (int32_t r = 0; r < count; r++) {
            dst_data[out_n++] = src_data[i];
        }
    }
    return out_n;
}

int64_t remora_replicate_f32(
    float* src_alloc, float* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* cnt_alloc, int32_t* cnt_align, int64_t cnt_off, int64_t cnt_n, int64_t cnt_str,
    float* dst_alloc, float* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)cnt_alloc; (void)cnt_n; (void)cnt_str;
    (void)dst_alloc; (void)dst_n; (void)dst_str;
    float*   src_data  = (float*)_mr_data(src_align, src_off);
    int32_t* cnt_data  = (int32_t*)_mr_data(cnt_align, cnt_off);
    float*   dst_data  = (float*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    int64_t out_n = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t count = cnt_data[i];
        for (int32_t r = 0; r < count; r++) {
            dst_data[out_n++] = src_data[i];
        }
    }
    return out_n;
}
