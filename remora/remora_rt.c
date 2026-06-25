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

/* MLIR runtime helper emitted by bufferization for memref copies.

   Signature matches MLIR's runner utility:
     memrefCopy(element_size_bytes, unranked_src, unranked_dst)

   Unranked memref descriptor:
     { int64_t rank; void *ranked_descriptor; }

   Ranked descriptor layout:
     allocated_ptr, aligned_ptr, offset, sizes[rank], strides[rank]
*/

typedef struct {
    int64_t rank;
    void* descriptor;
} _remora_unranked_memref_t;

typedef struct {
    void* allocated;
    void* aligned;
    int64_t offset;
    int64_t sizes_and_strides[];
} _remora_ranked_memref_t;

static void _remora_memref_copy_rec(
    int64_t elem_size,
    char* src_base,
    char* dst_base,
    const int64_t* sizes,
    const int64_t* src_strides,
    const int64_t* dst_strides,
    int64_t rank,
    int64_t dim
) {
    if (dim == rank) {
        memcpy(dst_base, src_base, (size_t)elem_size);
        return;
    }
    for (int64_t i = 0; i < sizes[dim]; i++) {
        _remora_memref_copy_rec(
            elem_size,
            src_base + i * src_strides[dim] * elem_size,
            dst_base + i * dst_strides[dim] * elem_size,
            sizes,
            src_strides,
            dst_strides,
            rank,
            dim + 1
        );
    }
}

void memrefCopy(int64_t elem_size, void* src_unranked, void* dst_unranked) {
    _remora_unranked_memref_t* src_ur = (_remora_unranked_memref_t*)src_unranked;
    _remora_unranked_memref_t* dst_ur = (_remora_unranked_memref_t*)dst_unranked;
    int64_t rank = src_ur->rank;
    _remora_ranked_memref_t* src = (_remora_ranked_memref_t*)src_ur->descriptor;
    _remora_ranked_memref_t* dst = (_remora_ranked_memref_t*)dst_ur->descriptor;

    char* src_base = (char*)src->aligned + src->offset * elem_size;
    char* dst_base = (char*)dst->aligned + dst->offset * elem_size;
    int64_t* sizes = src->sizes_and_strides;
    int64_t* src_strides = src->sizes_and_strides + rank;
    int64_t* dst_strides = dst->sizes_and_strides + rank;

    if (rank == 0) {
        memcpy(dst_base, src_base, (size_t)elem_size);
        return;
    }
    _remora_memref_copy_rec(
        elem_size,
        src_base,
        dst_base,
        sizes,
        src_strides,
        dst_strides,
        rank,
        0
    );
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

static int _cmp_f64_asc(const void* a, const void* b) {
    double va = *(const double*)a;
    double vb = *(const double*)b;
    return (va > vb) - (va < vb);
}

/* ── Sort (in-place) ────────────────────────────────────────────────────── */
/* LLVM ABI: (allocated, aligned, offset, size, stride) */

static void _remora_sort_i32_impl(int32_t* data, int64_t n) {
    qsort(data, (size_t)n, sizeof(int32_t), _cmp_i32_asc);
}

/* LSD radix sort for f32 (ascending).  Maps each float to a monotonic
   uint32 key (flip sign bit for positives, flip all bits for negatives),
   performs a 4-pass 8-bit radix sort, then maps the keys back to floats.
   Falls back to qsort for tiny inputs where setup overhead dominates. */
static inline uint32_t _f32_to_key(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline float _key_to_f32(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void _remora_radix_sort_f32(float* data, int64_t n) {
    if (n < 64) {
        qsort(data, (size_t)n, sizeof(float), _cmp_f32_asc);
        return;
    }
    uint32_t* src = (uint32_t*)malloc((size_t)n * sizeof(uint32_t));
    uint32_t* tmp = (uint32_t*)malloc((size_t)n * sizeof(uint32_t));
    if (!src || !tmp) {
        free(src);
        free(tmp);
        qsort(data, (size_t)n, sizeof(float), _cmp_f32_asc);
        return;
    }
    for (int64_t i = 0; i < n; i++) src[i] = _f32_to_key(data[i]);

    for (int shift = 0; shift < 32; shift += 8) {
        int64_t count[256];
        memset(count, 0, sizeof(count));
        for (int64_t i = 0; i < n; i++) {
            count[(src[i] >> shift) & 0xff]++;
        }
        int64_t total = 0;
        for (int b = 0; b < 256; b++) {
            int64_t c = count[b];
            count[b] = total;
            total += c;
        }
        for (int64_t i = 0; i < n; i++) {
            uint32_t key = src[i];
            tmp[count[(key >> shift) & 0xff]++] = key;
        }
        uint32_t* swap = src;
        src = tmp;
        tmp = swap;
    }

    for (int64_t i = 0; i < n; i++) data[i] = _key_to_f32(src[i]);
    free(src);
    free(tmp);
}

static void _remora_sort_f32_impl(float* data, int64_t n) {
    _remora_radix_sort_f32(data, n);
}

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
    _remora_radix_sort_f32((float*)_mr_data(aligned, offset), size);
}

void remora_sort_f64(
    double* allocated, double* aligned, int64_t offset, int64_t size, int64_t stride
) {
    (void)allocated; (void)stride;
    qsort(_mr_data(aligned, offset), (size_t)size, sizeof(double), _cmp_f64_asc);
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

static int _cmp_grade_f64(const void* a, const void* b) {
    const _grade_pair_t* ga = (const _grade_pair_t*)a;
    const _grade_pair_t* gb = (const _grade_pair_t*)b;
    double va = *(const double*)ga->base;
    double vb = *(const double*)gb->base;
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

void remora_grade_f64(
    double* src_alloc, double* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)dst_alloc; (void)dst_n; (void)dst_str;
    double*  src_data = (double*)_mr_data(src_align, src_off);
    int32_t* dst_data = (int32_t*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;

    _grade_pair_t* pairs = (_grade_pair_t*)malloc((size_t)n * sizeof(_grade_pair_t));
    for (int64_t i = 0; i < n; i++) {
        pairs[i].base = &src_data[i];
        pairs[i].index = (int)i;
    }
    qsort(pairs, (size_t)n, sizeof(_grade_pair_t), _cmp_grade_f64);
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

/* ── Replicate fill (void, for second phase after count) ───────────────── */

void remora_replicate_i32_fill(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* cnt_alloc, int32_t* cnt_align, int64_t cnt_off, int64_t cnt_n, int64_t cnt_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)cnt_alloc; (void)cnt_n; (void)cnt_str;
    (void)dst_alloc; (void)dst_str; (void)dst_n;
    int32_t* src_data = (int32_t*)_mr_data(src_align, src_off);
    int32_t* cnt_data = (int32_t*)_mr_data(cnt_align, cnt_off);
    int32_t* dst_data = (int32_t*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;
    int64_t out_n = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t count = cnt_data[i];
        for (int32_t r = 0; r < count; r++) {
            dst_data[out_n++] = src_data[i];
        }
    }
}

void remora_replicate_f32_fill(
    float* src_alloc, float* src_align, int64_t src_off, int64_t src_n, int64_t src_str,
    int32_t* cnt_alloc, int32_t* cnt_align, int64_t cnt_off, int64_t cnt_n, int64_t cnt_str,
    float* dst_alloc, float* dst_align, int64_t dst_off, int64_t dst_n, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)cnt_alloc; (void)cnt_n; (void)cnt_str;
    (void)dst_alloc; (void)dst_str; (void)dst_n;
    float*   src_data = (float*)_mr_data(src_align, src_off);
    int32_t* cnt_data = (int32_t*)_mr_data(cnt_align, cnt_off);
    float*   dst_data = (float*)_mr_data(dst_align, dst_off);
    int64_t n = src_n;
    int64_t out_n = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t count = cnt_data[i];
        for (int32_t r = 0; r < count; r++) {
            dst_data[out_n++] = src_data[i];
        }
    }
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

/* ── Matrix multiply (f32) ──────────────────────────────────────────────── */
/* LLVM ABI flattens each rank-2 memref<MxNxf32> to:
     (allocated, aligned, offset, size0, size1, stride0, stride1)
   Computes C = A * B with A: MxK, B: KxN, C: MxN (row-major). */

#ifdef REMORA_HAVE_BLAS
enum REMORA_CBLAS_ORDER { RemoraCblasRowMajor = 101 };
enum REMORA_CBLAS_TRANSPOSE { RemoraCblasNoTrans = 111 };
extern void cblas_sgemm(
    int order, int transa, int transb,
    int m, int n, int k,
    float alpha, const float* a, int lda,
    const float* b, int ldb,
    float beta, float* c, int ldc);
#endif

static void _remora_matmul_f32_fallback(
    const float* a, int64_t lda,
    const float* b, int64_t ldb,
    float* c, int64_t ldc,
    int64_t M, int64_t K, int64_t N
) {
    const int64_t BN = 256;
    const int64_t BK = 128;
    for (int64_t i = 0; i < M; i++) {
        float* crow = c + i * ldc;
        for (int64_t j = 0; j < N; j++) crow[j] = 0.0f;
    }
    for (int64_t jj = 0; jj < N; jj += BN) {
        int64_t jmax = jj + BN < N ? jj + BN : N;
        for (int64_t pp = 0; pp < K; pp += BK) {
            int64_t pmax = pp + BK < K ? pp + BK : K;
            int64_t i = 0;
            for (; i + 4 <= M; i += 4) {
                float* restrict c0 = c + (i + 0) * ldc;
                float* restrict c1 = c + (i + 1) * ldc;
                float* restrict c2 = c + (i + 2) * ldc;
                float* restrict c3 = c + (i + 3) * ldc;
                const float* restrict a0 = a + (i + 0) * lda;
                const float* restrict a1 = a + (i + 1) * lda;
                const float* restrict a2 = a + (i + 2) * lda;
                const float* restrict a3 = a + (i + 3) * lda;
                for (int64_t p = pp; p < pmax; p++) {
                    float v0 = a0[p], v1 = a1[p], v2 = a2[p], v3 = a3[p];
                    const float* restrict brow = b + p * ldb;
                    for (int64_t j = jj; j < jmax; j++) {
                        float bv = brow[j];
                        c0[j] += v0 * bv;
                        c1[j] += v1 * bv;
                        c2[j] += v2 * bv;
                        c3[j] += v3 * bv;
                    }
                }
            }
            for (; i < M; i++) {
                float* restrict crow = c + i * ldc;
                const float* restrict arow = a + i * lda;
                for (int64_t p = pp; p < pmax; p++) {
                    float aval = arow[p];
                    const float* restrict brow = b + p * ldb;
                    for (int64_t j = jj; j < jmax; j++) {
                        crow[j] += aval * brow[j];
                    }
                }
            }
        }
    }
}

void remora_matmul_f32(
    float* a_alloc, float* a_align, int64_t a_off, int64_t a_s0, int64_t a_s1, int64_t a_st0, int64_t a_st1,
    float* b_alloc, float* b_align, int64_t b_off, int64_t b_s0, int64_t b_s1, int64_t b_st0, int64_t b_st1,
    float* c_alloc, float* c_align, int64_t c_off, int64_t c_s0, int64_t c_s1, int64_t c_st0, int64_t c_st1
) {
    (void)a_alloc; (void)a_st1; (void)b_alloc; (void)b_s0; (void)b_st1;
    (void)c_alloc; (void)c_s0; (void)c_s1; (void)c_st1;
    float* a = (float*)_mr_data(a_align, a_off);
    float* b = (float*)_mr_data(b_align, b_off);
    float* c = (float*)_mr_data(c_align, c_off);
    int64_t M = a_s0, K = a_s1, N = b_s1;
    int64_t lda = a_st0, ldb = b_st0, ldc = c_st0;
#ifdef REMORA_HAVE_BLAS
    cblas_sgemm(
        RemoraCblasRowMajor, RemoraCblasNoTrans, RemoraCblasNoTrans,
        (int)M, (int)N, (int)K,
        1.0f, a, (int)lda, b, (int)ldb, 0.0f, c, (int)ldc);
#else
    _remora_matmul_f32_fallback(a, lda, b, ldb, c, ldc, M, K, N);
#endif
}

/* ── Per-row aliases for rank > 1 lowering ─────────────────────────────── */

void remora_sort_1d_i32(int32_t* a, int32_t* b, int64_t o, int64_t n, int64_t s) { remora_sort_i32(a, b, o, n, s); }
void remora_sort_1d_f32(float* a, float* b, int64_t o, int64_t n, int64_t s)   { remora_sort_f32(a, b, o, n, s); }
void remora_sort_1d_f64(double* a, double* b, int64_t o, int64_t n, int64_t s) { remora_sort_f64(a, b, o, n, s); }
void remora_grade_1d_i32(int32_t* sa, int32_t* sb, int64_t so, int64_t sn, int64_t ss, int32_t* da, int32_t* db, int64_t d_o, int64_t dn, int64_t ds) { remora_grade_i32(sa, sb, so, sn, ss, da, db, d_o, dn, ds); }
void remora_grade_1d_f32(float* sa, float* sb, int64_t so, int64_t sn, int64_t ss, int32_t* da, int32_t* db, int64_t d_o, int64_t dn, int64_t ds) { remora_grade_f32(sa, sb, so, sn, ss, da, db, d_o, dn, ds); }
void remora_grade_1d_f64(double* sa, double* sb, int64_t so, int64_t sn, int64_t ss, int32_t* da, int32_t* db, int64_t d_o, int64_t dn, int64_t ds) { remora_grade_f64(sa, sb, so, sn, ss, da, db, d_o, dn, ds); }

/* ── Scan (prefix sum) per-row helpers ──────────────────────────────────── */

void remora_scan_i32_1d(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t n, int64_t src_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dn, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)dst_alloc; (void)dst_str; (void)dn;
    int32_t* src = (int32_t*)_mr_data(src_align, src_off);
    int32_t* dst = (int32_t*)_mr_data(dst_align, dst_off);
    int32_t acc = 0;
    for (int64_t i = 0; i < n; i++) {
        acc += src[i];
        dst[i] = acc;
    }
}

void remora_scan_f32_1d(
    float* src_alloc, float* src_align, int64_t src_off, int64_t n, int64_t src_str,
    float* dst_alloc, float* dst_align, int64_t dst_off, int64_t dn, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)dst_alloc; (void)dst_str; (void)dn;
    float* src = (float*)_mr_data(src_align, src_off);
    float* dst = (float*)_mr_data(dst_align, dst_off);
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        acc += src[i];
        dst[i] = acc;
    }
}

/* ── Rotate per-row helpers ─────────────────────────────────────────────── */

void remora_rotate_i32_1d(
    int32_t* src_alloc, int32_t* src_align, int64_t src_off, int64_t n, int64_t src_str,
    int32_t* amt_alloc, int32_t* amt_align, int64_t amt_off, int64_t amt_n, int64_t amt_str,
    int32_t* dst_alloc, int32_t* dst_align, int64_t dst_off, int64_t dn, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)amt_alloc; (void)amt_n; (void)amt_str;
    (void)dst_alloc; (void)dst_str; (void)dn;
    int32_t* src = (int32_t*)_mr_data(src_align, src_off);
    int32_t* dst = (int32_t*)_mr_data(dst_align, dst_off);
    int32_t k = *(int32_t*)_mr_data(amt_align, amt_off);
    int32_t k_mod = ((k % (int32_t)n) + (int32_t)n) % (int32_t)n;
    for (int64_t i = 0; i < n; i++) {
        int64_t src_idx = (i + k_mod) % n;
        dst[i] = src[src_idx];
    }
}

void remora_rotate_f32_1d(
    float* src_alloc, float* src_align, int64_t src_off, int64_t n, int64_t src_str,
    int32_t* amt_alloc, int32_t* amt_align, int64_t amt_off, int64_t amt_n, int64_t amt_str,
    float* dst_alloc, float* dst_align, int64_t dst_off, int64_t dn, int64_t dst_str
) {
    (void)src_alloc; (void)src_str; (void)amt_alloc; (void)amt_n; (void)amt_str;
    (void)dst_alloc; (void)dst_str; (void)dn;
    float* src = (float*)_mr_data(src_align, src_off);
    float* dst = (float*)_mr_data(dst_align, dst_off);
    int32_t k = *(int32_t*)_mr_data(amt_align, amt_off);
    int32_t k_mod = ((k % (int32_t)n) + (int32_t)n) % (int32_t)n;
    for (int64_t i = 0; i < n; i++) {
        int64_t src_idx = (i + k_mod) % n;
        dst[i] = src[src_idx];
    }
}

/* ── Append per-row helpers ─────────────────────────────────────────────── */

void remora_append_i32_1d(
    int32_t* a_alloc, int32_t* a_align, int64_t a_off, int64_t a_n, int64_t a_str,
    int32_t* b_alloc, int32_t* b_align, int64_t b_off, int64_t b_n, int64_t b_str,
    int32_t* d_alloc, int32_t* d_align, int64_t d_off, int64_t d_n, int64_t d_str
) {
    (void)a_alloc; (void)a_str; (void)b_alloc; (void)b_str;
    (void)d_alloc; (void)d_str; (void)d_n;
    int32_t* a = (int32_t*)_mr_data(a_align, a_off);
    int32_t* b = (int32_t*)_mr_data(b_align, b_off);
    int32_t* d = (int32_t*)_mr_data(d_align, d_off);
    for (int64_t i = 0; i < a_n; i++) d[i] = a[i];
    for (int64_t i = 0; i < b_n; i++) d[a_n + i] = b[i];
}

void remora_append_f32_1d(
    float* a_alloc, float* a_align, int64_t a_off, int64_t a_n, int64_t a_str,
    float* b_alloc, float* b_align, int64_t b_off, int64_t b_n, int64_t b_str,
    float* d_alloc, float* d_align, int64_t d_off, int64_t d_n, int64_t d_str
) {
    (void)a_alloc; (void)a_str; (void)b_alloc; (void)b_str;
    (void)d_alloc; (void)d_str; (void)d_n;
    float* a = (float*)_mr_data(a_align, a_off);
    float* b = (float*)_mr_data(b_align, b_off);
    float* d = (float*)_mr_data(d_align, d_off);
    for (int64_t i = 0; i < a_n; i++) d[i] = a[i];
    for (int64_t i = 0; i < b_n; i++) d[a_n + i] = b[i];
}
