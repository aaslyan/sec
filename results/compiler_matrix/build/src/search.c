
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
