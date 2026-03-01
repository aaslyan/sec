
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
