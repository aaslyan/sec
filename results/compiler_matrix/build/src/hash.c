
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
