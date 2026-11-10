/*
* flatcar.c - Exact two-flatcar loader (1988 MCM Problem B)
*
*   1. Meet-in-the-middle: split crate types into groups A and B,
*      enumerate each group with a bounded DFS split across all threads.
*   2. Window: combine groups into single-car patterns with
*      length >= CAPL - delta (parallel over A entries, binary search in B).
*   3. Pairing: parallel over pattern a, shared incumbent, hard pruning.
*   4. Iterative widening: double delta until the result is provably optimal.
*
* Usage:
*   ./flatcar                 read from stdin (prompts if typing interactively)
*   ./flatcar -c FILE         read problem from a config file
*   ./flatcar -h              help
*
* Stdin format (whitespace-separated):
*   n
*   t_1 ... t_n     thickness in mm
*   w_1 ... w_n     weight in 1000 kg (tonnes)
*   c_1 ... c_n     number available
*   L  Wcap         car length (mm) and car weight limit (tonnes)
*   m  [S]          number of special crate types (the last m, 0 <= m < n);
*                   S = special limit in mm, both cars, read only if m > 0
*
* Config file format (key = value, '#' comments, lists space/comma separated):
*   thickness_mm     = ...      (required; n = number of entries)
*   weight_t         = ...      (required; n entries)
*   count            = ...      (required; n entries)
*   car_length_mm    = L        (required)
*   car_weight_t     = Wcap     (required)
*   special_count    = m        (optional, default 0, 0 <= m < n)
*   special_limit_mm = S        (required if m > 0)
*
* Build:  gcc -O2 -fopenmp flatcar.c -o flatcar
* Run:    OMP_NUM_THREADS=8 ./flatcar -c flatcar.cfg
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <limits.h>
#include <omp.h>
 
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif
 
#define MAXN   16
#define SCALE  10       /* lengths stored internally in 0.1 mm units */
#define WSCALE 1000     /* weights stored internally in kg           */
#define MAX_PRINT 20    /* max optimal solutions to print            */
 
/* ---------------- Problem parameters (read from input) ---------------- */
static double CAR_LEN_MM;      /* usable length per car (mm)        */
static double CAR_WT_T;        /* weight limit per car (tonnes)     */
static double SPEC_LEN_MM;     /* special-crate limit, both cars    */
static int    m_special;       /* last m crate types are special    */
static double RAW_T[MAXN];     /* thickness as entered (mm)         */
static double RAW_W[MAXN];     /* weight as entered (tonnes)        */
 
static int interactive = 0;    /* 1 if prompting on a terminal      */
 
/* ---------------- Data structures ---------------- */
typedef struct {
    int len;                    /* 0.1 mm                    */
    int wt;                     /* kg                        */
    int spec;                   /* 0.1 mm of special crates  */
    unsigned short cnt[MAXN];
} Load;
 
typedef struct { Load *v; size_t n, cap; } Vec;
typedef struct { long i, j; } Pair;
 
static int n;
static int T[MAXN], W[MAXN], N[MAXN], isSpec[MAXN];
static int CAPL, CAPW, CAPS;
 
static int    best;             /* shared incumbent: best total length */
static Pair  *sols  = NULL;
static size_t nsols = 0, capsols = 0;
 
/* ---------------- Utilities ---------------- */
static long round_nonneg(double x) { return (long)(x + 0.5); }
 
static void prompt(const char *fmt, ...)
{
    if (!interactive) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}
 
static void push(Vec *a, const Load *x)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? 2 * a->cap : 1024;
        a->v = realloc(a->v, a->cap * sizeof(Load));
        if (!a->v) { perror("realloc"); exit(1); }
    }
    a->v[a->n++] = *x;
}
 
static int cmp_len_asc(const void *p, const void *q)
{
    const Load *a = p, *b = q;
    return (a->len > b->len) - (a->len < b->len);
}
 
/* longest first; ties broken by counts so output is deterministic */
static int cmp_len_desc(const void *p, const void *q)
{
    const Load *a = p, *b = q;
    if (a->len != b->len) return (a->len < b->len) - (a->len > b->len);
    return memcmp(a->cnt, b->cnt, sizeof a->cnt);
}
 
static int cmp_pair(const void *p, const void *q)
{
    const Pair *a = p, *b = q;
    if (a->i != b->i) return (a->i > b->i) - (a->i < b->i);
    return (a->j > b->j) - (a->j < b->j);
}
 
static size_t lower_bound(const Vec *B, int key)   /* first len >= key */
{
    size_t lo = 0, hi = B->n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (B->v[mid].len < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}
 
static size_t upper_bound(const Vec *B, int key)   /* first len > key */
{
    size_t lo = 0, hi = B->n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (B->v[mid].len <= key) lo = mid + 1; else hi = mid;
    }
    return lo;
}
 
/* ---------------- Input: stdin (interactive or redirected) ---------------- */
static int read_stdin(void)
{
    interactive = isatty(fileno(stdin));
    prompt("=== Two-Flatcar Loading Problem ===\n\n");
 
    prompt("Number of crate types n (1-%d): ", MAXN);
    if (scanf("%d", &n) != 1 || n < 1 || n > MAXN) {
        fprintf(stderr, "Need 1 <= n <= %d\n", MAXN); return 0;
    }
 
    prompt("\nEnter the THICKNESS of each crate type in millimeters (mm).\n");
    for (int i = 0; i < n; i++) {
        prompt("  C%d thickness (mm): ", i + 1);
        if (scanf("%lf", &RAW_T[i]) != 1 || RAW_T[i] < 0) {
            fprintf(stderr, "Bad thickness %d\n", i + 1); return 0;
        }
    }
 
    prompt("\nEnter the WEIGHT of each crate type in metric tons "
           "(1 t = 1000 kg; decimals OK, e.g. 0.5).\n");
    for (int i = 0; i < n; i++) {
        prompt("  C%d weight (t): ", i + 1);
        if (scanf("%lf", &RAW_W[i]) != 1 || RAW_W[i] < 0) {
            fprintf(stderr, "Bad weight %d\n", i + 1); return 0;
        }
    }
 
    prompt("\nEnter the NUMBER AVAILABLE of each crate type "
           "(whole number, 0-65535).\n");
    for (int i = 0; i < n; i++) {
        prompt("  C%d count: ", i + 1);
        if (scanf("%d", &N[i]) != 1 || N[i] < 0 || N[i] > 65535) {
            fprintf(stderr, "Bad count %d\n", i + 1); return 0;
        }
    }
 
    prompt("\nBoth flatcars are assumed identical.\n");
    prompt("  Usable length of each car (mm): ");
    if (scanf("%lf", &CAR_LEN_MM) != 1 || CAR_LEN_MM <= 0) {
        fprintf(stderr, "Need a positive car length (mm)\n"); return 0;
    }
    prompt("  Weight limit of each car (t): ");
    if (scanf("%lf", &CAR_WT_T) != 1 || CAR_WT_T <= 0) {
        fprintf(stderr, "Need a positive car weight limit (t)\n"); return 0;
    }
 
    prompt("\nSpecial restriction: the LAST m crate types share a combined\n"
           "thickness limit across both cars. Enter 0 for no restriction.\n");
    prompt("  Number of special crate types m (0-%d): ", n - 1);
    if (scanf("%d", &m_special) != 1) {
        fprintf(stderr, "Missing number of special crate types m\n");
        return 0;
    }
    if (m_special < 0 || m_special >= n) {
        fprintf(stderr, "m must satisfy 0 <= m < n (got m = %d, n = %d)\n",
                m_special, n);
        return 0;
    }
    if (m_special > 0) {
        if (m_special == 1)
            prompt("  Special crate type: C%d\n", n);
        else
            prompt("  Special crate types: C%d-C%d\n", n - m_special + 1, n);
        prompt("  Combined thickness limit for special crates, "
               "both cars together (mm): ");
        if (scanf("%lf", &SPEC_LEN_MM) != 1 || SPEC_LEN_MM < 0) {
            fprintf(stderr, "Need a non-negative special limit (mm)\n");
            return 0;
        }
    }
    prompt("\n");
    return 1;
}
 
/* ---------------- Input: config file ---------------- */
static char *trim(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) e--;
    *e = '\0';
    return s;
}
 
/* parse space/comma separated numbers; returns count, -1 too many, -2 bad */
static int parse_list(const char *s, double *out, int maxn)
{
    int k = 0;
    for (;;) {
        while (isspace((unsigned char)*s) || *s == ',') s++;
        if (!*s) break;
        if (k >= maxn) return -1;
        char *end;
        double v = strtod(s, &end);
        if (end == s) return -2;
        out[k++] = v;
        s = end;
    }
    return k;
}
 
static int is_whole(double v) { return v == (double)(long)v; }
 
static int read_config(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 0; }
 
    char line[4096];
    int lineno = 0;
    int nt = -1, nw = -1, nc = -1;
    int haveL = 0, haveW = 0, haveM = 0, haveS = 0;
    double cbuf[MAXN], mval = 0;
 
    while (fgets(line, sizeof line, f)) {
        lineno++;
        if (!strchr(line, '\n') && !feof(f)) {
            fprintf(stderr, "%s:%d: line too long\n", path, lineno);
            goto fail;
        }
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *s = trim(line);
        if (!*s) continue;
 
        char *eq = strchr(s, '=');
        if (!eq) {
            fprintf(stderr, "%s:%d: expected 'key = value'\n", path, lineno);
            goto fail;
        }
        *eq = '\0';
        char *key = trim(s);
        char *val = trim(eq + 1);
 
        double v[MAXN];
        int k = parse_list(val, v, MAXN);
        if (k == -1) {
            fprintf(stderr, "%s:%d: too many values for '%s' (max %d)\n",
                    path, lineno, key, MAXN);
            goto fail;
        }
        if (k < 0) {
            fprintf(stderr, "%s:%d: bad number in '%s'\n", path, lineno, key);
            goto fail;
        }
        if (k == 0) {
            fprintf(stderr, "%s:%d: no value for '%s'\n", path, lineno, key);
            goto fail;
        }
 
        int scalar_ok = (k == 1);
        if (!strcmp(key, "thickness_mm")) {
            memcpy(RAW_T, v, k * sizeof(double)); nt = k;
        } else if (!strcmp(key, "weight_t")) {
            memcpy(RAW_W, v, k * sizeof(double)); nw = k;
        } else if (!strcmp(key, "count")) {
            memcpy(cbuf, v, k * sizeof(double)); nc = k;
        } else if (!strcmp(key, "car_length_mm")) {
            if (!scalar_ok) goto need_scalar;
            CAR_LEN_MM = v[0]; haveL = 1;
        } else if (!strcmp(key, "car_weight_t")) {
            if (!scalar_ok) goto need_scalar;
            CAR_WT_T = v[0]; haveW = 1;
        } else if (!strcmp(key, "special_count")) {
            if (!scalar_ok) goto need_scalar;
            mval = v[0]; haveM = 1;
        } else if (!strcmp(key, "special_limit_mm")) {
            if (!scalar_ok) goto need_scalar;
            SPEC_LEN_MM = v[0]; haveS = 1;
        } else {
            fprintf(stderr, "%s:%d: unknown key '%s'\n", path, lineno, key);
            goto fail;
        }
        continue;
 
    need_scalar:
        fprintf(stderr, "%s:%d: '%s' takes a single value\n", path, lineno, key);
        goto fail;
    }
    fclose(f);
 
    /* ---- required keys and consistency ---- */
    if (nt < 0 || nw < 0 || nc < 0) {
        fprintf(stderr, "%s: need thickness_mm, weight_t and count\n", path);
        return 0;
    }
    if (nw != nt || nc != nt) {
        fprintf(stderr, "%s: thickness_mm, weight_t and count must have the "
                "same number of entries (got %d, %d, %d)\n", path, nt, nw, nc);
        return 0;
    }
    if (!haveL || !haveW) {
        fprintf(stderr, "%s: need car_length_mm and car_weight_t\n", path);
        return 0;
    }
    n = nt;
 
    for (int i = 0; i < n; i++) {
        if (RAW_T[i] < 0) {
            fprintf(stderr, "%s: C%d thickness must be >= 0\n", path, i + 1);
            return 0;
        }
        if (RAW_W[i] < 0) {
            fprintf(stderr, "%s: C%d weight must be >= 0\n", path, i + 1);
            return 0;
        }
        if (!is_whole(cbuf[i]) || cbuf[i] < 0 || cbuf[i] > 65535) {
            fprintf(stderr, "%s: C%d count must be a whole number 0-65535\n",
                    path, i + 1);
            return 0;
        }
        N[i] = (int)cbuf[i];
    }
    if (CAR_LEN_MM <= 0 || CAR_WT_T <= 0) {
        fprintf(stderr, "%s: car length and weight limit must be positive\n",
                path);
        return 0;
    }
 
    m_special = 0;
    if (haveM) {
        if (!is_whole(mval) || mval < 0 || mval >= n) {
            fprintf(stderr, "%s: special_count must be a whole number with "
                    "0 <= m < n (n = %d)\n", path, n);
            return 0;
        }
        m_special = (int)mval;
    }
    if (m_special > 0) {
        if (!haveS) {
            fprintf(stderr, "%s: special_limit_mm is required when "
                    "special_count > 0\n", path);
            return 0;
        }
        if (SPEC_LEN_MM < 0) {
            fprintf(stderr, "%s: special_limit_mm must be >= 0\n", path);
            return 0;
        }
    } else if (haveS) {
        fprintf(stderr, "Note: special_limit_mm ignored (special_count = 0)\n");
    }
    return 1;
 
fail:
    fclose(f);
    return 0;
}
 
/* ---------------- Step 1: enumerate one group (bounded DFS) ---------------- */
static void enum_half(const int *idx, int m, int k, Load *cur, Vec *out)
{
    if (k == m) { push(out, cur); return; }
 
    int i = idx[k];
    int maxc = N[i];
    if (T[i] > 0) { int b = (CAPL - cur->len) / T[i]; if (b < maxc) maxc = b; }
    if (W[i] > 0) { int b = (CAPW - cur->wt)  / W[i]; if (b < maxc) maxc = b; }
    if (isSpec[i] && T[i] > 0) {
        int b = (CAPS - cur->spec) / T[i]; if (b < maxc) maxc = b;
    }
 
    for (int c = 0; c <= maxc; c++) {
        cur->cnt[i] = (unsigned short)c;
        enum_half(idx, m, k + 1, cur, out);
        cur->len += T[i];
        cur->wt  += W[i];
        if (isSpec[i]) cur->spec += T[i];
    }
    /* undo the (maxc + 1) additions */
    cur->len -= (maxc + 1) * T[i];
    cur->wt  -= (maxc + 1) * W[i];
    if (isSpec[i]) cur->spec -= (maxc + 1) * T[i];
    cur->cnt[i] = 0;
}
 
/* Enumerate one group with all threads: split the DFS on its first types. */
static void enum_half_par(const int *idx, int m, Vec *out)
{
    int nthreads = omp_get_max_threads();
 
    /* choose prefix depth so there are enough tasks to balance the load */
    int depth = 0;
    double nprefix = 1.0;
    while (depth < m && nprefix < 8.0 * nthreads) {
        nprefix *= N[idx[depth]] + 1;
        depth++;
    }
 
    /* feasible prefixes over the first `depth` types (cheap, serial) */
    Vec prefixes = {0};
    Load z;
    memset(&z, 0, sizeof z);
    enum_half(idx, depth, 0, &z, &prefixes);
 
    /* finish each prefix's subtree in parallel */
    #pragma omp parallel
    {
        Vec loc = {0};
 
        #pragma omp for schedule(dynamic, 1)
        for (long p = 0; p < (long)prefixes.n; p++) {
            Load cur = prefixes.v[p];
            enum_half(idx, m, depth, &cur, &loc);
        }
 
        #pragma omp critical(merge_half)
        {
            for (size_t k = 0; k < loc.n; k++) push(out, &loc.v[k]);
        }
        free(loc.v);
    }
    free(prefixes.v);
}
 
/* ---------------- Step 2: single-car patterns with len >= lo ---------------- */
static Vec window_patterns(const Vec *A, const Vec *B, int lo)
{
    Vec P = {0};
 
    #pragma omp parallel
    {
        Vec loc = {0};
 
        #pragma omp for schedule(dynamic, 16)
        for (long ia = 0; ia < (long)A->n; ia++) {
            const Load *a = &A->v[ia];
            size_t j0 = lower_bound(B, lo - a->len);
            size_t j1 = upper_bound(B, CAPL - a->len);
 
            for (size_t j = j0; j < j1; j++) {
                const Load *b = &B->v[j];
                if (a->wt + b->wt > CAPW)     continue;
                if (a->spec + b->spec > CAPS) continue;
 
                Load p;
                memset(&p, 0, sizeof p);
                p.len  = a->len  + b->len;
                p.wt   = a->wt   + b->wt;
                p.spec = a->spec + b->spec;
                for (int k = 0; k < n; k++)
                    p.cnt[k] = (unsigned short)(a->cnt[k] + b->cnt[k]);
                push(&loc, &p);
            }
        }
 
        #pragma omp critical(merge)
        {
            for (size_t k = 0; k < loc.n; k++) push(&P, &loc.v[k]);
        }
        free(loc.v);
    }
    return P;
}
 
/* ---------------- Step 3: pairing ---------------- */
static int compatible(const Load *a, const Load *b)
{
    if (a->spec + b->spec > CAPS) return 0;
    for (int k = 0; k < n; k++)
        if (a->cnt[k] + b->cnt[k] > N[k]) return 0;
    return 1;
}
 
static void record(long i, long j)
{
    if (nsols == capsols) {
        capsols = capsols ? 2 * capsols : 64;
        sols = realloc(sols, capsols * sizeof(Pair));
        if (!sols) { perror("realloc"); exit(1); }
    }
    sols[nsols].i = i;
    sols[nsols].j = j;
    nsols++;
}
 
/* P must be sorted by length, longest first. Pairs (i, j) with j >= i,
* so each unordered pair of car loads is considered exactly once.     */
static void pair_patterns(const Vec *P)
{
    best  = -1;
    nsols = 0;
 
    #pragma omp parallel for schedule(dynamic, 32)
    for (long i = 0; i < (long)P->n; i++) {
        const Load *a = &P->v[i];
        int cur;
 
        #pragma omp atomic read
        cur = best;
        if (2 * a->len < cur) continue;   /* every partner is <= a in length */
 
        for (long j = i; j < (long)P->n; j++) {
            const Load *b = &P->v[j];
            int tot = a->len + b->len;
 
            #pragma omp atomic read
            cur = best;
            if (tot < cur) break;         /* sorted: totals only decrease */
 
            if (!compatible(a, b)) continue;
 
            #pragma omp critical(incumbent)
            {
                if (tot > best) { best = tot; nsols = 0; }
                if (tot == best) record(i, j);
            }
        }
    }
}
 
/* ---------------- Output ---------------- */
static void print_load(const char *name, const Load *x)
{
    printf("    %s:", name);
    for (int k = 0; k < n; k++) printf(" C%d=%-2d", k + 1, x->cnt[k]);
    printf(" | %8.1f mm  %7.3f t\n",
           x->len / (double)SCALE, x->wt / (double)WSCALE);
}
 
static void report(const Vec *P)
{
    qsort(sols, nsols, sizeof(Pair), cmp_pair);
 
    printf("\nOptimal loaded length: %.1f mm of %.1f mm\n",
           best / (double)SCALE, 2 * CAPL / (double)SCALE);
    printf("Wasted floor space:    %.1f mm (%.2f cm)\n",
           (2 * CAPL - best) / (double)SCALE,
           (2 * CAPL - best) / (double)SCALE / 10.0);
    printf("Distinct optimal loadings (cars interchangeable): %zu\n", nsols);
 
    for (size_t s = 0; s < nsols && s < MAX_PRINT; s++) {
        const Load *a = &P->v[sols[s].i];
        const Load *b = &P->v[sols[s].j];
        if (m_special > 0)
            printf("\n  Solution %zu  (special crates: %.1f mm of %.1f mm)\n",
                   s + 1, (a->spec + b->spec) / (double)SCALE,
                   CAPS / (double)SCALE);
        else
            printf("\n  Solution %zu\n", s + 1);
        print_load("Car 1", a);
        print_load("Car 2", b);
    }
    if (nsols > MAX_PRINT)
        printf("\n  ... %zu more not shown\n", nsols - MAX_PRINT);
}
 
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s              read input from stdin "
            "(prompts if interactive)\n"
            "       %s -c FILE      read problem from a config file\n"
            "       %s -h           show this help\n",
            prog, prog, prog);
}
 
/* ---------------- Main ---------------- */
int main(int argc, char **argv)
{
    const char *cfg = NULL;
 
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-c") || !strcmp(argv[a], "--config")) {
            if (a + 1 >= argc) { usage(argv[0]); return 1; }
            cfg = argv[++a];
        } else if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[a]);
            usage(argv[0]);
            return 1;
        }
    }
 
    if (cfg) {
        if (!read_config(cfg)) return 1;
        printf("Config: %s\n", cfg);
    } else {
        if (!read_stdin()) return 1;
    }
 
    /* guard against int overflow in the internal units */
    const double LIM = INT_MAX / 4.0;
    if (CAR_LEN_MM * SCALE > LIM || CAR_WT_T * WSCALE > LIM ||
        (m_special > 0 && SPEC_LEN_MM * SCALE > LIM)) {
        fprintf(stderr, "Capacity too large for internal integer units\n");
        return 1;
    }
    for (int i = 0; i < n; i++)
        if (RAW_T[i] * SCALE > LIM || RAW_W[i] * WSCALE > LIM) {
            fprintf(stderr, "Crate %d thickness/weight too large\n", i + 1);
            return 1;
        }
 
    /* ---- integer units: 0.1 mm and kg ---- */
    for (int i = 0; i < n; i++) {
        T[i] = (int)round_nonneg(RAW_T[i] * SCALE);
        W[i] = (int)round_nonneg(RAW_W[i] * WSCALE);
        isSpec[i] = (i >= n - m_special);   /* last m types are special */
    }
    CAPL = (int)round_nonneg(CAR_LEN_MM * SCALE);
    CAPW = (int)round_nonneg(CAR_WT_T   * WSCALE);
    CAPS = (m_special > 0) ? (int)round_nonneg(SPEC_LEN_MM * SCALE)
                           : INT_MAX / 4;   /* effectively no limit */
 
    /* ---- echo the problem ---- */
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Crates:\n");
    for (int i = 0; i < n; i++)
        printf("  C%-2d %8.1f mm  %7.3f t  x %-5d%s\n",
               i + 1, RAW_T[i], RAW_W[i], N[i],
               isSpec[i] ? "  (special)" : "");
    printf("Car: %.1f mm, %.3f t each\n", CAR_LEN_MM, CAR_WT_T);
    if (m_special > 0)
        printf("Special: C%d-C%d, limit %.1f mm total\n",
               n - m_special + 1, n, SPEC_LEN_MM);
    else
        printf("Special: none\n");
 
    /* ---- split types: special types all go to B, the rest balance sizes ---- */
    int idxA[MAXN], idxB[MAXN], mA = 0, mB = 0;
    double sizeA = 1.0, sizeB = 1.0;
    int order[MAXN], no = 0;
 
    for (int i = 0; i < n; i++) {
        if (isSpec[i]) { idxB[mB++] = i; sizeB *= N[i] + 1; }
        else order[no++] = i;
    }
    for (int x = 0; x < no; x++)            /* sort non-special by count, desc */
        for (int y = x + 1; y < no; y++)
            if (N[order[y]] > N[order[x]]) {
                int tmp = order[x]; order[x] = order[y]; order[y] = tmp;
            }
    for (int x = 0; x < no; x++) {
        int i = order[x];
        if (sizeA <= sizeB) { idxA[mA++] = i; sizeA *= N[i] + 1; }
        else                { idxB[mB++] = i; sizeB *= N[i] + 1; }
    }
 
    printf("Group A:"); for (int k = 0; k < mA; k++) printf(" C%d", idxA[k] + 1);
    printf("   Group B:"); for (int k = 0; k < mB; k++) printf(" C%d", idxB[k] + 1);
    printf("\n");
 
    /* ---- Step 1: enumerate both groups, each using all threads ---- */
    double t0 = omp_get_wtime();
    Vec A = {0}, B = {0};
 
    enum_half_par(idxA, mA, &A);
    enum_half_par(idxB, mB, &B);
    qsort(B.v, B.n, sizeof(Load), cmp_len_asc);
    printf("Group sizes: |A| = %zu, |B| = %zu  (%.4f s)\n",
           A.n, B.n, omp_get_wtime() - t0);
 
    /* ---- Steps 2-4: iterative widening ---- */
    int delta = 1 * SCALE;                  /* start with 1 mm slack */
    for (;;) {
        int lo = CAPL - delta;
        if (lo < 0) lo = 0;
 
        double t1 = omp_get_wtime();
        Vec P = window_patterns(&A, &B, lo);
        qsort(P.v, P.n, sizeof(Load), cmp_len_desc);
        double t2 = omp_get_wtime();
        pair_patterns(&P);
        double t3 = omp_get_wtime();
 
        printf("Window >= %8.1f mm: %9zu patterns  gen %.4f s  pair %.4f s  ",
               lo / (double)SCALE, P.n, t2 - t1, t3 - t2);
        if (best >= 0) printf("best total %.1f mm\n", best / (double)SCALE);
        else           printf("no feasible pair\n");
 
        /* optimal if any better pair would have both cars inside the window */
        if ((best >= 0 && best >= 2 * CAPL - delta) || lo == 0) {
            report(&P);
            free(P.v);
            break;
        }
        free(P.v);
        if (delta > INT_MAX / 2) delta = CAPL;   /* avoid overflow when doubling */
        else delta *= 2;
    }
 
    printf("\nTotal time: %.4f s\n", omp_get_wtime() - t0);
    free(A.v); free(B.v); free(sols);
    return 0;
}
