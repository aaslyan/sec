
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
