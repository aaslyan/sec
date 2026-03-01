
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
