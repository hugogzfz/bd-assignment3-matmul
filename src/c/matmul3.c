/*
 * Assignment 3 - sequential and OpenMP-parallel blocked dense multiplication (plus the A1 ijk reference).
 * Usage: matmul3 <seq|omp|ijk> <n> <reps> <warmup> <batch> <A> <B> <dumpC|-> <bs> <threads>
 * The blocked kernel is mul_blocked of bd-assignment2-matmul @ 1aab86d, restricted to a range of rows of C.
 * Every element of C adds a_ik*b_kj for k increasing (no FMA), so all methods and thread counts return identical bits.
 */
#define _DARWIN_C_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <pthread.h>
#include <omp.h>

#define MAX_THREADS 256

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void *xmalloc(size_t bytes, const char *what) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "out of memory (%s)\n", what); exit(3); }
    return p;
}

static double *read_dense(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); exit(2); }
    double *m = xmalloc(n * n * sizeof(double), "dense input");
    if (fread(m, sizeof(double), n * n, f) != n * n) {
        fprintf(stderr, "short read on %s (expected %zu doubles)\n", path, n * n);
        exit(2);
    }
    fclose(f);
    return m;
}

/* Current resident set size and number of OS threads of this process (Mach task info). */
static long resident_bytes(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS)
        return (long)info.resident_size;
#endif
    return -1;
}

static int os_thread_count(void) {
#ifdef __APPLE__
    thread_act_array_t list;
    mach_msg_type_number_t cnt;
    if (task_threads(mach_task_self(), &list, &cnt) != KERN_SUCCESS) return -1;
    for (mach_msg_type_number_t i = 0; i < cnt; i++) mach_port_deallocate(mach_task_self(), list[i]);
    vm_deallocate(mach_task_self(), (vm_address_t)list, cnt * sizeof(thread_act_t));
    return (int)cnt;
#else
    return -1;
#endif
}

/* Reference only (Assignment 1, unchanged): i-j-k with a scalar accumulator. */
__attribute__((noinline))
static void mul_ijk(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/* A2 blocked kernel on rows [r0, r1) of C. With r0 = 0, r1 = n it is exactly the A2 kernel. */
__attribute__((noinline))
static void mul_blocked_rows(const double *restrict a, const double *restrict b, double *restrict c,
                             size_t n, size_t bs, size_t r0, size_t r1) {
    if (r1 > r0) memset(c + r0 * n, 0, (r1 - r0) * n * sizeof(double));
    for (size_t ii = r0; ii < r1; ii += bs) {
        size_t imax = ii + bs < r1 ? ii + bs : r1;
        for (size_t kk = 0; kk < n; kk += bs) {
            size_t kmax = kk + bs < n ? kk + bs : n;
            for (size_t jj = 0; jj < n; jj += bs) {
                size_t jmax = jj + bs < n ? jj + bs : n;
                for (size_t i = ii; i < imax; i++) {
                    double *restrict ci = c + i * n;
                    for (size_t k = kk; k < kmax; k++) {
                        const double aik = a[i * n + k];
                        const double *restrict bk = b + k * n;
                        for (size_t j = jj; j < jmax; j++) {
                            ci[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }
}

/* Parallel version: thread t of q owns the contiguous rows [n*t/q, n*(t+1)/q) of C, so no two threads
 * write the same element and no synchronisation is needed apart from the implicit barrier at the end. */
static int team_size[MAX_THREADS];      /* omp_get_num_threads() seen by each thread */
static int had_rows[MAX_THREADS];       /* 1 if the thread received at least one row */
static uint64_t os_tid[MAX_THREADS];    /* OS thread id that executed each OpenMP thread */

__attribute__((noinline))
static void mul_blocked_omp(const double *restrict a, const double *restrict b, double *restrict c,
                            size_t n, size_t bs, int threads) {
    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), q = omp_get_num_threads();
        size_t r0 = n * (size_t)t / (size_t)q, r1 = n * (size_t)(t + 1) / (size_t)q;
        mul_blocked_rows(a, b, c, n, bs, r0, r1);
        if (t < MAX_THREADS) { team_size[t] = q; had_rows[t] = r1 > r0; pthread_threadid_np(NULL, &os_tid[t]); }
    }
}

/* Plain left-to-right sum of all elements in row-major order. */
static double checksum_dense(const double *c, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n * n; i++) s += c[i];
    return s;
}

int main(int argc, char **argv) {
    if (argc < 11) {
        fprintf(stderr, "usage: %s seq|omp|ijk n reps warmup batch A B dumpC|- bs threads\n", argv[0]);
        return 1;
    }
    const char *method = argv[1];
    size_t n = strtoull(argv[2], NULL, 10);
    int reps = atoi(argv[3]), warmup = atoi(argv[4]);
    long batch = atol(argv[5]);
    if (batch < 1) batch = 1;
    const char *dump = strcmp(argv[8], "-") ? argv[8] : NULL;
    size_t bs = strtoull(argv[9], NULL, 10);
    int threads = atoi(argv[10]);
    int m_seq = !strcmp(method, "seq"), m_omp = !strcmp(method, "omp"), m_ijk = !strcmp(method, "ijk");
    if (!m_seq && !m_omp && !m_ijk) { fprintf(stderr, "unknown method %s\n", method); return 1; }
    if (bs < 1 || threads < 1 || threads > MAX_THREADS) { fprintf(stderr, "need bs >= 1 and 1 <= threads <= %d\n", MAX_THREADS); return 1; }
    if (!m_omp) threads = 1;     /* seq and ijk never enter an OpenMP region */

    printf("META method=%s n=%zu reps=%d warmup=%d batch=%ld bs=%zu threads=%d\n", method, n, reps, warmup, batch, bs, threads);
    double *a = read_dense(argv[6], n);
    double *b = read_dense(argv[7], n);
    double *c = xmalloc(n * n * sizeof(double), "dense C");
    memset(c, 0, n * n * sizeof(double));    /* touch pages before timing */
    printf("RSS_AFTER_LOAD_BYTES %ld\nOS_THREADS_BEFORE %d\n", resident_bytes(), os_thread_count());
    fflush(stdout);

    for (int w = 0; w < warmup + reps; w++) {
        int64_t t0 = now_ns();
        for (long q = 0; q < batch; q++) {
            if (m_seq) mul_blocked_rows(a, b, c, n, bs, 0, n);
            else if (m_omp) mul_blocked_omp(a, b, c, n, bs, threads);
            else mul_ijk(a, b, c, n);
        }
        int64_t t1 = now_ns();
        printf("%s %d %.9e %.17g\n", w < warmup ? "WARMUP" : "REP", w < warmup ? w : w - warmup,
               (double)(t1 - t0) * 1e-9 / (double)batch, checksum_dense(c, n));
        fflush(stdout);
    }

    if (m_omp) {      /* actual team size of the last call and how many threads received work */
        int team = team_size[0], working = 0, distinct = 0;
        for (int t = 0; t < team && t < MAX_THREADS; t++) {
            working += had_rows[t];
            int seen = 0;
            for (int u = 0; u < t; u++) seen |= os_tid[u] == os_tid[t];
            distinct += !seen;
        }
        printf("TEAM_SIZE %d\nWORKING_THREADS %d\nDISTINCT_OS_THREADS %d\n", team, working, distinct);
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    /* ru_maxrss is in bytes on macOS (kilobytes on Linux) */
    printf("OS_THREADS_AFTER %d\nMAXRSS_BYTES %ld\n", os_thread_count(), (long)ru.ru_maxrss);
    fflush(stdout);
    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f || fwrite(c, sizeof(double), n * n, f) != n * n) { fprintf(stderr, "cannot write dump\n"); return 4; }
        fclose(f);
    }
    printf("DONE\n");
    return 0;
}
