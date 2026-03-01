"""
Compiler matrix experiment: 5 C projects × {gcc,clang} × {O0,O2,O3}.

Builds 30 binaries in a temp directory, disassembles them into Binary
objects with category=project and compiler="{cc}-{opt}", then runs the
full analysis pipeline plus compiler-specific metrics.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from extraction.disassemble import disassemble_binary
from utils.helpers import Binary, save_json, save_pickle, ensure_output_dir

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# C source programs
# ─────────────────────────────────────────────────────────────────────────────

SOURCES: Dict[str, str] = {

"sort": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

static void median3(int *arr, int lo, int mid, int hi) {
    if (arr[lo] > arr[mid]) swap(&arr[lo], &arr[mid]);
    if (arr[lo] > arr[hi])  swap(&arr[lo], &arr[hi]);
    if (arr[mid] > arr[hi]) swap(&arr[mid], &arr[hi]);
}

static int partition(int *arr, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    median3(arr, lo, mid, hi);
    int pivot = arr[mid];
    swap(&arr[mid], &arr[hi - 1]);
    int i = lo, j = hi - 1;
    for (;;) {
        while (arr[++i] < pivot) {}
        while (arr[--j] > pivot) {}
        if (i >= j) break;
        swap(&arr[i], &arr[j]);
    }
    swap(&arr[i], &arr[hi - 1]);
    return i;
}

static void qsort_r2(int *arr, int lo, int hi) {
    while (lo < hi) {
        if (hi - lo < 16) {
            for (int i = lo + 1; i <= hi; i++) {
                int key = arr[i], j = i - 1;
                while (j >= lo && arr[j] > key) { arr[j+1] = arr[j]; j--; }
                arr[j+1] = key;
            }
            break;
        }
        int p = partition(arr, lo, hi);
        if (p - lo < hi - p) { qsort_r2(arr, lo, p - 1); lo = p + 1; }
        else                  { qsort_r2(arr, p + 1, hi); hi = p - 1; }
    }
}

static int cmp_int(const void *a, const void *b) {
    return (*(int*)a > *(int*)b) - (*(int*)a < *(int*)b);
}

static int is_sorted(const int *a, int n) {
    for (int i = 1; i < n; i++) if (a[i] < a[i-1]) return 0;
    return 1;
}

static void fill_rand(int *a, int n, unsigned seed) {
    for (int i = 0; i < n; i++) { seed = seed * 1664525u + 1013904223u; a[i] = (int)seed; }
}

int main(void) {
    int sizes[] = {100, 1000, 5000, 20000};
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        int *a = malloc(n * sizeof(int));
        int *b = malloc(n * sizeof(int));
        fill_rand(a, n, 42u + (unsigned)n);
        memcpy(b, a, n * sizeof(int));
        qsort_r2(a, 0, n - 1);
        qsort(b, n, sizeof(int), cmp_int);
        int ok = is_sorted(a, n) && memcmp(a, b, n * sizeof(int)) == 0;
        printf("n=%-6d sorted=%d match=%d\n", n, is_sorted(a, n), ok);
        free(a); free(b);
    }
    return 0;
}
""",

"hash": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NBUCKETS 4096
#define FNV_BASIS UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x00000100000001b3)

typedef struct Node { char key[48]; uint64_t val; struct Node *next; } Node;
static Node *table[NBUCKETS];

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = FNV_BASIS;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= FNV_PRIME; }
    return h;
}

static void ht_put(const char *k, uint64_t v) {
    uint64_t h = fnv1a(k, strlen(k));
    Node **slot = &table[h % NBUCKETS];
    for (Node *n = *slot; n; n = n->next)
        if (strcmp(n->key, k) == 0) { n->val = v; return; }
    Node *n = calloc(1, sizeof(Node));
    strncpy(n->key, k, 47);
    n->val = v; n->next = *slot; *slot = n;
}

static uint64_t ht_get(const char *k) {
    uint64_t h = fnv1a(k, strlen(k));
    for (Node *n = table[h % NBUCKETS]; n; n = n->next)
        if (strcmp(n->key, k) == 0) return n->val;
    return UINT64_MAX;
}

static void ht_del(const char *k) {
    uint64_t h = fnv1a(k, strlen(k));
    Node **p = &table[h % NBUCKETS];
    while (*p) {
        if (strcmp((*p)->key, k) == 0) { Node *tmp = *p; *p = tmp->next; free(tmp); return; }
        p = &(*p)->next;
    }
}

int main(void) {
    char buf[48];
    for (int i = 0; i < 2000; i++) { snprintf(buf, sizeof(buf), "k%05d", i); ht_put(buf, (uint64_t)i * 6364136223846793005ULL); }
    uint64_t s = 0;
    for (int i = 0; i < 2000; i++) { snprintf(buf, sizeof(buf), "k%05d", i); s ^= ht_get(buf); }
    for (int i = 0; i < 1000; i++) { snprintf(buf, sizeof(buf), "k%05d", i * 2); ht_del(buf); }
    uint64_t s2 = 0;
    for (int i = 0; i < 2000; i++) { snprintf(buf, sizeof(buf), "k%05d", i); uint64_t v = ht_get(buf); if (v != UINT64_MAX) s2 ^= v; }
    printf("xor_all=0x%016lx xor_odd=0x%016lx\n", s, s2);
    return 0;
}
""",

"compress": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- RLE ---- */
typedef struct { uint8_t run; uint8_t byte; } Pair;

static size_t rle_enc(const uint8_t *in, size_t n, Pair *out) {
    size_t o = 0;
    for (size_t i = 0; i < n; ) {
        uint8_t b = in[i]; uint8_t r = 1;
        while (i+r < n && in[i+r] == b && r < 255) r++;
        out[o++] = (Pair){r, b}; i += r;
    }
    return o;
}
static size_t rle_dec(const Pair *p, size_t np, uint8_t *out) {
    size_t o = 0;
    for (size_t i = 0; i < np; i++) for (uint8_t r = 0; r < p[i].run; r++) out[o++] = p[i].byte;
    return o;
}

/* ---- LZ77 sliding-window (simplified) ---- */
typedef struct { uint16_t off; uint8_t len; uint8_t lit; } LZToken;

static size_t lz77_enc(const uint8_t *in, size_t n, LZToken *out) {
    size_t o = 0, i = 0;
    while (i < n) {
        int best_off = 0, best_len = 0;
        size_t win = i > 255 ? i - 255 : 0;
        for (size_t j = win; j < i; j++) {
            size_t l = 0;
            while (l < 15 && i+l < n && in[j+l] == in[i+l]) l++;
            if ((int)l > best_len) { best_len = (int)l; best_off = (int)(i - j); }
        }
        if (best_len >= 3) {
            out[o++] = (LZToken){(uint16_t)best_off, (uint8_t)best_len, 0};
            i += best_len;
        } else {
            out[o++] = (LZToken){0, 0, in[i++]};
        }
    }
    return o;
}

static void make_data(uint8_t *buf, size_t n) {
    uint32_t s = 0xdeadbeef;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        /* 60% repetition: repeat previous run */
        buf[i] = (s & 0xff) < 153 && i > 0 ? buf[i - 1 - (s % (i < 8 ? i : 8))] : (uint8_t)(s >> 8);
    }
}

int main(void) {
    size_t N = 32768;
    uint8_t *orig = malloc(N), *dec = malloc(N);
    Pair  *rle_buf  = malloc(N * sizeof(Pair));
    LZToken *lz_buf = malloc(N * sizeof(LZToken));

    make_data(orig, N);

    size_t rp = rle_enc(orig, N, rle_buf);
    size_t rd = rle_dec(rle_buf, rp, dec);
    printf("RLE: %zu->%zu pairs (%.2fx) roundtrip=%d\n",
           N, rp, (double)N / (rp * 2), rd == N && !memcmp(orig, dec, N));

    size_t lt = lz77_enc(orig, N, lz_buf);
    printf("LZ77: %zu tokens (%.2fx)\n", lt, (double)N / (lt * 4));

    free(orig); free(dec); free(rle_buf); free(lz_buf);
    return 0;
}
""",

"search": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Brute force ---- */
static int bf(const char *t, int tn, const char *p, int pm) {
    int c = 0;
    for (int i = 0; i <= tn - pm; i++) {
        int j; for (j = 0; j < pm; j++) if (t[i+j] != p[j]) break;
        if (j == pm) c++;
    }
    return c;
}

/* ---- KMP ---- */
static void kmp_fail(const char *p, int m, int *f) {
    f[0] = 0; int k = 0;
    for (int i = 1; i < m; i++) {
        while (k && p[k] != p[i]) k = f[k-1];
        if (p[k] == p[i]) k++;
        f[i] = k;
    }
}
static int kmp(const char *t, int tn, const char *p, int pm) {
    int *f = malloc(pm * sizeof(int));
    kmp_fail(p, pm, f);
    int c = 0, q = 0;
    for (int i = 0; i < tn; i++) {
        while (q && p[q] != t[i]) q = f[q-1];
        if (p[q] == t[i]) q++;
        if (q == pm) { c++; q = f[q-1]; }
    }
    free(f); return c;
}

/* ---- Boyer-Moore-Horspool ---- */
static int bmh(const char *t, int tn, const char *p, int pm) {
    if (pm == 0) return tn + 1;
    int skip[256];
    for (int i = 0; i < 256; i++) skip[i] = pm;
    for (int i = 0; i < pm - 1; i++) skip[(uint8_t)p[i]] = pm - 1 - i;
    int c = 0, i = 0;
    while (i <= tn - pm) {
        int j = pm - 1;
        while (j >= 0 && p[j] == t[i+j]) j--;
        if (j < 0) { c++; i++; } else i += skip[(uint8_t)t[i+pm-1]];
    }
    return c;
}

static void gen_text(char *buf, int n) {
    const char alpha[] = "abcdefgh";
    uint32_t s = 0x1234abcd;
    for (int i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        buf[i] = alpha[s % 8];
    }
    buf[n] = '\0';
}

int main(void) {
    int N = 80000;
    char *text = malloc(N + 1);
    gen_text(text, N);
    const char *pats[] = {"abc", "abcd", "abcde", "bcdefgh", "abcdefgh"};
    for (int p = 0; p < 5; p++) {
        const char *pat = pats[p];
        int pm = (int)strlen(pat);
        int r1 = bf(text, N, pat, pm);
        int r2 = kmp(text, N, pat, pm);
        int r3 = bmh(text, N, pat, pm);
        printf("'%s' bf=%d kmp=%d bmh=%d %s\n", pat, r1, r2, r3, (r1==r2 && r2==r3)?"OK":"MISMATCH");
    }
    free(text);
    return 0;
}
""",

"matrix": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SZ 80

typedef double M[SZ][SZ];

static void fill(M m, double a, double b) {
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            m[i][j] = sin(a*i + b*j) + cos(b*i - a*j);
}
static void zero(M m) { memset(m, 0, sizeof(M)); }

static void mul(const M a, const M b, M c) {
    zero(c);
    for (int i = 0; i < SZ; i++)
        for (int k = 0; k < SZ; k++) {
            double aik = a[i][k];
            for (int j = 0; j < SZ; j++)
                c[i][j] += aik * b[k][j];
        }
}
static void transpose(const M a, M b) {
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            b[j][i] = a[i][j];
}
static double frobenius(const M m) {
    double s = 0;
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            s += m[i][j] * m[i][j];
    return sqrt(s);
}
static void add(const M a, const M b, M c) {
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            c[i][j] = a[i][j] + b[i][j];
}
static void scale(M m, double f) {
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            m[i][j] *= f;
}

int main(void) {
    M *a = malloc(sizeof(M)), *b = malloc(sizeof(M));
    M *c = malloc(sizeof(M)), *at = malloc(sizeof(M));
    M *d = malloc(sizeof(M)), *e = malloc(sizeof(M));

    fill(*a, 0.1, 0.2);
    fill(*b, 0.3, 0.4);

    mul(*a, *b, *c);
    transpose(*a, *at);
    mul(*at, *b, *d);
    add(*c, *d, *e);
    scale(*e, 0.5);

    printf("||A||=%.4f ||B||=%.4f ||C||=%.4f ||D||=%.4f ||(C+D)/2||=%.4f\n",
           frobenius(*a), frobenius(*b), frobenius(*c), frobenius(*d), frobenius(*e));

    free(a); free(b); free(c); free(at); free(d); free(e);
    return 0;
}
""",
}

CONFIGS: List[Tuple[str, str, str]] = [
    ("gcc",   "-O0", "gcc-O0"),
    ("gcc",   "-O2", "gcc-O2"),
    ("gcc",   "-O3", "gcc-O3"),
    ("clang", "-O0", "clang-O0"),
    ("clang", "-O2", "clang-O2"),
    ("clang", "-O3", "clang-O3"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Build + disassemble
# ─────────────────────────────────────────────────────────────────────────────

def _compile(src_path: Path, out_path: Path, cc: str, opt: str) -> bool:
    """Compile src_path → out_path.  Returns True on success."""
    extra = ["-lm"] if "matrix" in src_path.name else []
    cmd = [cc, opt, "-g0", str(src_path), "-o", str(out_path)] + extra
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            logger.warning(f"Compile failed ({' '.join(cmd)}): {r.stderr[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Compile error: {e}")
        return False


def build_corpus(build_dir: Path) -> List[Binary]:
    """Write C sources, compile all 30 variants, disassemble into Binary objects."""
    src_dir = build_dir / "src"
    bin_dir = build_dir / "bin"
    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Write sources
    for name, code in SOURCES.items():
        (src_dir / f"{name}.c").write_text(code)

    binaries: List[Binary] = []
    for project, code in SOURCES.items():
        src_path = src_dir / f"{project}.c"
        for cc, opt, label in CONFIGS:
            bin_name = f"{project}_{label}"
            out_path = bin_dir / bin_name
            logger.info(f"Compiling {bin_name}  ({cc} {opt})")
            if not _compile(src_path, out_path, cc, opt):
                continue
            binary = disassemble_binary(out_path, category=project, compiler=label)
            if binary is None:
                logger.warning(f"Disassembly failed for {bin_name}")
                continue
            # Override name to include full label
            binary.name = bin_name
            binaries.append(binary)
            logger.info(
                f"  {bin_name}: {binary.function_count} funcs, "
                f"{binary.instruction_count} insns"
            )

    logger.info(f"Built {len(binaries)} / {len(SOURCES) * len(CONFIGS)} binaries")
    return binaries


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_compiler_matrix(args) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_root   = ensure_output_dir(Path(args.output_dir))
    corpus_dir = ensure_output_dir(out_root / "corpus")
    results_dir = ensure_output_dir(out_root / "results")
    build_dir  = out_root / "build"

    # ── 1. Build ──────────────────────────────────────────────────────────
    binaries = build_corpus(build_dir)
    if not binaries:
        logger.error("No binaries produced — check compiler installation")
        return 1

    save_pickle(binaries, corpus_dir / "corpus.pkl")
    save_json(
        [{"name": b.name, "project": b.category, "compiler": b.compiler,
          "functions": b.function_count, "instructions": b.instruction_count}
         for b in binaries],
        corpus_dir / "corpus_summary.json",
    )
    logger.info(f"Corpus saved: {len(binaries)} binaries")

    # ── 2. Core analyses ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")

    from analysis.frequency   import run_frequency_analysis
    from analysis.ngrams      import run_ngram_analysis
    from analysis.compression import run_compression_analysis
    from clustering.ncd       import run_ncd_analysis
    from clustering.similarity import run_ngram_similarity_analysis

    run_frequency_analysis(binaries, results_dir)
    run_ngram_analysis(binaries, results_dir)
    run_compression_analysis(binaries, results_dir)
    run_ncd_analysis(binaries, results_dir)
    run_ngram_similarity_analysis(binaries, results_dir)

    # ── 3. Compiler-matrix-specific analysis ──────────────────────────────
    from experiments.compiler_matrix_analysis import run_compiler_matrix_analysis
    run_compiler_matrix_analysis(binaries, results_dir)

    logger.info(f"All results written to {results_dir}")
    return 0
