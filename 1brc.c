/*
 * One Billion Row Challenge (1BRC)
 *
 * Input:  a text file of lines "<city>;<temperature>\n", e.g. "Hamburg;12.3".
 *         - Temperature is always one of: X.X, XX.X, -X.X, -XX.X
 *           (i.e. a value in [-99.9, 99.9] with exactly one decimal place).
 *         - City names are UTF-8 byte strings with no ';' or '\n' in them,
 *           and are between 1 and 100 bytes long.
 * Output: "{city=min/mean/max, city=min/mean/max, ...}\n", cities sorted by
 *         name, each statistic rounded to one decimal place.
 *
 * High-level strategy
 * -------------------
 *   1. The file is split into roughly equal byte ranges, one per worker.
 *   2. Workers are spawned as a *tree of forked processes* (fan-out up to 8 per
 *      level) so that the fork/merge cost scales with core count. Each leaf
 *      process owns one byte range; interior processes only fork children and
 *      merge their results.
 *   3. Each worker mmaps its range and parses it 8 lines at a time with AVX2,
 *      accumulating per-city {sum, count, min, max} into a custom open-addressing
 *      hash table.
 *   4. Results bubble back up the process tree via pipes + shared memory and are
 *      merged. The root sorts the final city list and prints it.
 *
 * Temperatures are handled as fixed-point integers in tenths of a degree
 * (e.g. "12.3" -> 123), so MIN_TEMP/MAX_TEMP are -999/999.
 *
 * Notes on data layout
 * --------------------
 *   - "Short" cities (name fits in a 32-byte AVX2 register, i.e. <= 31 bytes)
 *     live inline in the hash table as a single __m256i.
 *   - "Long" cities (32..100 bytes) are stored out-of-line in a separate table;
 *     the inline slot then holds a LONG_CITY_SENTINEL + an index reference.
 *   - Each city slot has STRIDE(=8) independent accumulators (one per SIMD
 *     lane) to avoid cross-lane reduction each iteration; they are summed once
 *     at the end in convert_hash_to_results().
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <sched.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <x86intrin.h>

/* ========================================================================== *
 *  Compile-time configuration
 * ========================================================================== */

/*
 * wait() for all child processes before exiting.
 * If false, "cheat" by returning immediately, leaving orphan processes for the
 * OS to reap.
 */
#define UNMAP false

/*
 * Pin worker threads to a CPU.
 * Very helpful at high core counts, slightly harmful at low ones.
 *
 * Each worker is assigned to the lowest unused CPU # we're scheduled for.
 * For hyperthreaded CPUs, logical CPU numbers must not be interleaved.
 */
#define PIN_CPU true

/*
 * Print timing information and a city summary to stderr.
 */
#define DEBUG false

/* ========================================================================== *
 *  Sizes and tuning constants
 * ========================================================================== */

/* Hash table index widths (in bits). */
#define HASH_SHIFT        17 /* short-city table: 17 balances small/large inputs
                              * (16 is 1% faster on small / 10% slower on 10k;
                              *  18 is 1% slower on small / 3% faster on 10k). */
#define HASH_LONG_SHIFT   14 /* long-city table: smallest width that fits 10k. */
#define HASH_RESULT_SHIFT 14 /* results table: smallest width that fits 10k. */

#define MAX_CITIES 10000
#define MAX_TEMP   999  /* +99.9 degrees, in tenths */
#define MIN_TEMP   -999 /* -99.9 degrees, in tenths */

#define SHORT_CITY_LENGTH 32  /* one AVX2 register */
#define LONG_CITY_LENGTH  128 /* four AVX2 registers */

/* Power-of-two table sizes. */
#define HASH_ENTRIES      (1 << HASH_SHIFT)
#define HASH_LONG_ENTRIES (1 << HASH_LONG_SHIFT)

/* 32-byte AVX2 registers fit 8 values at once; STRIDE == lanes processed per
 * iteration. Moving to AVX-512/SSE (64/16 bytes) would also need code changes. */
#define STRIDE 8

/* ========================================================================== *
 *  Type definitions
 * ========================================================================== */

/* Pointers into a worker's hash arena (all carved from one mmap). */
typedef struct {
  int * const restrict packedOffsets;      /* offsets of occupied short slots, insertion order */
  void * const restrict hashedCities;      /* SHORT_CITY_LENGTH bytes per slot: the city name */
  void * const restrict hashedStorage;     /* HASH_ENTRY_SIZE bytes per slot: STRIDE accumulators */
  void * const restrict hashedCitiesLong;  /* LONG_CITY_LENGTH bytes per long-city slot */
} HashPointers;

typedef struct {
  int numCities;
  int numCitiesLong;
} HashCounts;

typedef struct {
  const HashPointers p;
  HashCounts counts;
} Hash;

/* One unit of work: a byte range [start, end) of the file. */
typedef struct {
  long start;
  long end;
  int fd;
  int workerId;
  int cpuId;
  bool fork;
  bool warmup;
  bool first; /* owns the very first byte of the file */
  bool last;  /* owns the very last byte of the file */
} Worker;

/*
 * One per-lane accumulator for a city. min/max are tracked as (min, -max) so a
 * single signed-min operation can update both.
 * 'sum' and 'count' are bit-packed together into packedSumCount (see SUM_BITS).
 */
typedef struct {
  int64_t packedSumCount;
  int32_t min;
  int32_t negmax;
} HashEntry;

/* When a city is "long", its inline 32-byte slot instead holds this reference:
 * a sentinel tag plus an index into the long-city table. */
typedef struct {
  int32_t sentinel;
  int32_t index;
  int32_t padding[6];
} LongCityRef;

typedef union {
  __m256i reg;
  char bytes[SHORT_CITY_LENGTH];
} ShortCity;

typedef union {
  __m256i regs[4];
  char bytes[LONG_CITY_LENGTH];
} LongCity;

/* A city as stored inline in a table slot: either a short name (ShortCity) or a
 * long-city reference (LongCityRef), distinguished by city_is_long(). */
typedef union {
  __m256i reg;
  ShortCity shortCity;
  LongCityRef longRef;
} PackedCity;

/* Final per-city result, after the STRIDE lane accumulators have been summed. */
typedef struct {
  PackedCity city;
  int64_t sum;
  int32_t count;
  int16_t min;
  int16_t max;
} ResultsRow;

/* Reference to a ResultsRow by byte offset into the results 'rows' table. */
typedef struct {
  uint32_t offset;
} ResultsRef;

typedef struct {
  int numCities;
  int numLongCities;
  ResultsRef * restrict refs;       /* one per city, used for sorting/printing */
  ResultsRow * restrict rows;        /* open-addressing table of results */
  LongCity * restrict longCities;    /* out-of-line long city names */
} Results;

/* ========================================================================== *
 *  Forward declarations
 * ========================================================================== */

/* Orchestration */
void prep_workers(Worker *workers, int numWorkers, bool warmup, int fd, struct stat *fileStat);
void process(int id, Worker *workers, int numWorkers, int fd, Results *out);
void start_worker(Worker *w, Results *out);

/* Parsing / accumulation main logic */
void process_chunk(const void * const restrict base, const uint32_t *offsets, Hash * restrict h);
__m256i process_long(const void * const restrict start, Hash * restrict h, int * restrict semicolonBytesOut);

/* Hashing */
inline __m256i hash_cities(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h);
inline int hash_city(__m256i str);
int hash_long(long x, long y);

/* Insertion */
inline long insert_city(Hash * restrict h, long hash, const __m256i maskedCity);
int insert_city_long1(Hash * restrict h, int hash, __m256i seg0, __m256i seg1);
int insert_city_long2(Hash * restrict h, int hash, __m256i seg0, __m256i seg1, __m256i seg2);
int insert_city_long3(Hash * restrict h, int hash, __m256i seg0, __m256i seg1, __m256i seg2, __m256i seg3);

/* Reduction / output */
void convert_hash_to_results(Hash * restrict hash, Results * restrict out);
void merge(Results * restrict dst, Results * restrict src);
int sort_result(const void *a, const void *b, void *arg);
void print_results(Results *results);

/* Small helpers */
inline void setup_results(Results *r);
uint32_t find_next_row(const void *data, uint32_t offset);
inline __m256i city_from_long_hash(int hashValue);
inline bool city_is_long(PackedCity city);

/* Debug */
void debug_results(Results *results);
void print256(__m256i var);

/* ========================================================================== *
 *  Debug / timing macros
 * ========================================================================== */

#if DEBUG
#define D(x) x
#define TIMER_RESET()  clock_gettime(CLOCK_MONOTONIC, &tic);
#define TIMER_MS(name) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-12s: %9.3f ms\n", name, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000);
#define TIMER_MS_NUM(name, n) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-9s %2d: %9.3f ms\n", name, n, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000);
#define TIMER_US(name)
#define TIMER_INIT()   struct timespec tic, toc; (void)tic; (void)toc; TIMER_RESET();
#else
#define D(x)
#define TIMER_RESET()
#define TIMER_INIT()
#define TIMER_MS_NUM(name, n)
#define TIMER_MS(name)
#define TIMER_US(name)
#endif

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

/* Branchless integer min/max.
 * NOTE: these double-evaluate both arguments.
 */
#define MIN(x, y) (y + ((x - y) & ((x - y) >> 31)))
#define MAX(x, y) (x - ((x - y) & ((x - y) >> 31)))

/* ========================================================================== *
 *  Page / cache-line arithmetic
 * ========================================================================== */

#define PAGE_SIZE            0X1000
#define HUGE_PAGE_SIZE       0x200000
#define PAGE_MASK            (~(PAGE_SIZE      - 1))
#define HUGE_PAGE_MASK       (~(HUGE_PAGE_SIZE - 1))
#define PAGE_TRUNC(v)        ((v) & (PAGE_MASK))
#define HUGE_PAGE_TRUNC(v)   ((v) & (HUGE_PAGE_MASK))
#define PAGE_CEIL(v)         (PAGE_TRUNC(v      + PAGE_SIZE      - 1))
#define HUGE_PAGE_CEIL(v)    (HUGE_PAGE_TRUNC(v + HUGE_PAGE_SIZE - 1))
#define PAGE_TRUNC_P(p)      ((void *)PAGE_TRUNC((uintptr_t)p))
#define HUGE_PAGE_TRUNC_P(p) ((void *)HUGE_PAGE_TRUNC((uintptr_t)p))
#define PAGE_CEIL_P(p)       ((void *)PAGE_CEIL((uintptr_t)p))
#define HUGE_PAGE_CEIL_P(p)  ((void *)HUGE_PAGE_CEIL((uintptr_t)p))

#define LINE_SIZE 64
#define LINE_MASK (~(LINE_SIZE - 1))
#define LINE_TRUNC(v) ((v) & (LINE_MASK))
#define LINE_CEIL(v)  (LINE_TRUNC(v + LINE_SIZE - 1))

/* ========================================================================== *
 *  Hash index arithmetic
 *
 *  An integer 'hash' is used as a byte offset into hashedCities, (stride of SHORT_CITY_LENGTH).
 *  The same value, scaled, indexes hashedStorage (stride of HASH_ENTRY_SIZE == STRIDE * HashEntry).
 * ========================================================================== */

#define HASH_ENTRY_SIZE ((int)(STRIDE * sizeof(HashEntry))) /* 8 * 16 = 128 bytes */

#define HASH_DATA_OFFSET 5        /* log2(SHORT_CITY_LENGTH) */
#define HASH_CITY_OFFSET 5        /* log2(SHORT_CITY_LENGTH) */
#define HASH_CITY_LONG_OFFSET 7   /* log2(LONG_CITY_LENGTH)  */

#define HASH_SHORT_MASK  (((1 << HASH_SHIFT       ) - 1) << MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))
#define HASH_LONG_MASK   (((1 << HASH_LONG_SHIFT  ) - 1) << HASH_CITY_LONG_OFFSET)
#define HASH_RESULT_MASK (((1 << HASH_RESULT_SHIFT) - 1) << HASH_CITY_OFFSET)

#define HASH_DATA_SHIFT (HASH_DATA_OFFSET - MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))
#define HASH_CITY_SHIFT (HASH_CITY_OFFSET - MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))

/* ========================================================================== *
 *  Arena sizes
 * ========================================================================== */

#define PACKED_OFFSETS_SIZE     PAGE_CEIL((int)sizeof(int) * MAX_CITIES)
#define HASHED_CITIES_SIZE      HUGE_PAGE_CEIL(SHORT_CITY_LENGTH * HASH_ENTRIES)
#define HASHED_DATA_SIZE        HUGE_PAGE_CEIL(HASH_ENTRY_SIZE   * HASH_ENTRIES)
#define HASHED_CITIES_LONG_SIZE HUGE_PAGE_CEIL(LONG_CITY_LENGTH  * HASH_LONG_ENTRIES)

#define HASH_MEMORY_SIZE (PACKED_OFFSETS_SIZE + HASHED_CITIES_SIZE + HASHED_DATA_SIZE + HASHED_CITIES_LONG_SIZE)

#define RESULTS_SIZE             LINE_CEIL(sizeof(Results))
#define RESULTS_REFS_SIZE        LINE_CEIL(sizeof(ResultsRef)  * MAX_CITIES)
#define RESULTS_ROWS_SIZE        LINE_CEIL(sizeof(ResultsRow)  * HASH_ENTRIES)
#define RESULTS_LONG_CITIES_SIZE LINE_CEIL(sizeof(LongCity)    * MAX_CITIES)

#define RESULTS_MEMORY_SIZE PAGE_CEIL(RESULTS_SIZE + RESULTS_REFS_SIZE + RESULTS_ROWS_SIZE + RESULTS_LONG_CITIES_SIZE)

/* Per-worker file window. One 4 GiB mapping reused chunk-by-chunk: a small dummy
 * region up front, the file mapped in the middle, and a trailing buffer so the
 * SIMD loads near the end never fault. */
#define MMAP_DATA_SIZE (1L << 32)
#define DUMMY_SIZE     PAGE_SIZE
#define TRAILING_SPACE PAGE_SIZE
#define MAX_CHUNK_SIZE (MMAP_DATA_SIZE - DUMMY_SIZE - TRAILING_SPACE)

/* ========================================================================== *
 *  sum/count bit-packing
 *
 *  Each HashEntry packs a running sum and count into one int64:
 *    bits [0 .. SUM_BITS]                = sum + SUM_SIGN_BIT (bias to stay >= 0)
 *    bits [COUNT_BITS_START .. 63]       = count
 *  SUM_BITS is sized so the per-lane sum (at most 1e9 * 999 / 8 / 8) can't
 *  overflow into the count field.
 * ========================================================================== */

#define SUM_BITS         35 /* 1 + ceil(log2(1B * 999 / 8 / 8)) */
#define SUM_SIGN_BIT     (1L << (SUM_BITS))
#define COUNT_BITS_START (SUM_BITS + 1)

#define EXTRACT_COUNT(v) ((int)(v >> COUNT_BITS_START))
#define SUM_MASK         ((1L << COUNT_BITS_START) - 1)
#define EXTRACT_SUM(v)   ((v & SUM_MASK) - SUM_SIGN_BIT)

/*
 * Low null byte guarantees this is an invalid city name.
 */
#define LONG_CITY_SENTINEL 0xFACADE00

/* ========================================================================== *
 *  Global constant data
 * ========================================================================== */

/* The dummy line "\0AD;0.0\n" that finished lanes are pointed at so they keep
 * parsing harmless input. The leading '\0' guarantees it never collides with a
 * real city name, and its accumulator slot is never enumerated into results. */
alignas(32) const long MASKED_DUMMY_DATA[] = {
  'A' << 8 | 'D' << 16, 0, 0, 0
};
const void * const MASKED_DUMMY = MASKED_DUMMY_DATA;

/* 32 bytes of 0xFF followed by 32 bytes of 0x00. Loading at
 * (CITY_MASK + 32 - n) yields a mask with n leading 0xFF bytes, used to keep
 * the first n bytes of a city name and zero the rest. */
alignas(64) const long CITY_MASK_DATA[] = {
    -1, -1, -1, -1, 0, 0, 0, 0,
};
const void * const CITY_MASK = CITY_MASK_DATA;

/* ========================================================================== *
 *  main
 * ========================================================================== */

int main(int argc, char** argv) {
  TIMER_INIT();

  if (argc < 3 || argc > 4) {
    fprintf(stderr, "Usage: 1brc file workers [warmup]\n");
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    perror("Error opening file");
    return EXIT_FAILURE;
  }

  struct stat fileStat;
  if (fstat(fd, &fileStat) == -1) {
    perror("Error getting file size");
    close(fd);
    return EXIT_FAILURE;
  }

  int numWorkers = atoi(argv[2]);
  if (numWorkers < 1 || numWorkers > 256) {
    fprintf(stderr, "workers must be between 1 and 256\n");
    return EXIT_FAILURE;
  }

  const bool warmup = DEBUG && (argc < 4 ? false : atoi(argv[3]) != 0);

  /* Don't spawn more workers than there are pages of work. */
  if ((fileStat.st_size - 1) / PAGE_SIZE < numWorkers) {
    D(fprintf(stderr, "decreasing numWorkers to %ld\n", fileStat.st_size / PAGE_SIZE + 1);)
    numWorkers = (int) (fileStat.st_size / PAGE_SIZE) + 1;
  }

  /* One mmap holds the Results struct followed by the Worker array. */
  void *mem = mmap(NULL, RESULTS_MEMORY_SIZE + sizeof(Worker) * numWorkers, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Results *results = mem;
  setup_results(results);

  mem += RESULTS_MEMORY_SIZE;

  Worker *workers = mem;
  prep_workers(workers, numWorkers, warmup, fd, &fileStat);

  /* Fast path: a single worker with no process-tree machinery. */
  TIMER_RESET();
  if (UNMAP && numWorkers == 1) {
    start_worker(workers, results);
  }
  else {
    process(0, workers, numWorkers, -1, results);
  }
  TIMER_MS("process");

  TIMER_RESET();
  qsort_r(results->refs, results->numCities, sizeof(ResultsRef), sort_result, results);
  TIMER_US("sort");

  TIMER_RESET();
  print_results(results);
  TIMER_US("print");

  D(debug_results(results));
  return 0;
}

/* ========================================================================== *
 *  Worker setup
 * ========================================================================== */

/* Carve the file into numWorkers page-aligned byte ranges and assign each to a
 * CPU we're allowed to run on. */
void prep_workers(Worker *workers, int numWorkers, bool warmup, int fd, struct stat *fileStat) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);
  int numCpus = CPU_COUNT(&cpuset);

  if (numCpus < numWorkers) {
    fprintf(stderr, "%d threads is less than %d available CPUS\n", numWorkers, numCpus);
    exit(1);
  }

  long cpu = 0;
  long start = 0;
  long delta = PAGE_TRUNC(fileStat->st_size / numWorkers);
  for (int i = 0; i < numWorkers; i++) {
    /* Pick the next CPU from our affinity mask. */
    while (!CPU_ISSET(cpu, &cpuset)) {
      cpu++;
    }

    Worker *w = workers + i;
    w->workerId = i;
    w->cpuId = cpu++;
    w->fd = fd;
    w->start = start;
    w->end = (start += delta);
    w->first = i == 0;
    w->last = i == numWorkers - 1;
    if (w->last) {
      w->end = fileStat->st_size; /* last worker absorbs the page-alignment remainder */
    }
    w->warmup = warmup;
  }
}

/* Lay out the sub-arrays of a Results arena (refs, rows, longCities) directly
 * after the Results city counts. */
void setup_results(Results *r) {
  r->numCities = 0;
  r->numLongCities = 0;

  void *p = (void *)r;
  p += RESULTS_SIZE;

  r->refs = p;
  p += RESULTS_REFS_SIZE;

  r->rows = p;
  p += RESULTS_ROWS_SIZE;

  r->longCities = p;
  p += RESULTS_LONG_CITIES_SIZE;
}

/* ========================================================================== *
 *  Process tree: fork children, merge their results
 * ========================================================================== */

/*
 * Recursively fan out work across forked processes.
 *
 *   - If numWorkers <= max_k, this node forks every child worker process;
 *     each child is a leaf that calls start_worker() and writes its results
 *     to shared memory, finishing with signalling completion over a pipe.
 *   - Otherwise this node forks k interior children processes, each of which
 *     recurses into process() with a slice of the workers, building a tree of
 *     depth ceil(log_k(numWorkers)).
 *
 * After forking, this node polls the pipes and merge()s each child's results
 * into 'out' as it finishes. If fdOut != -1 we are ourselves an interior node
 * and signal our parent when done.
 */
void process(int id, Worker *workers, int numWorkers, int fdOut, Results *out) {
  TIMER_INIT();

  const int max_k = 8;
  const bool doWork = numWorkers <= max_k;      /* children will be leaves, not interior nodes */
  const int k = doWork ? numWorkers : (numWorkers + (max_k - 1)) / max_k;

  int fd[k][2];
  struct pollfd poll_fds[k];

  /* Shared memory for children to write their Results into. */
  void *tmp = mmap(NULL, RESULTS_MEMORY_SIZE * k, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  Results *childResults[k];
  for (int i = 0; i < k; i++) {
    childResults[i] = tmp + i * RESULTS_MEMORY_SIZE;
    setup_results(childResults[i]);
  }

  int new_id = id;
  for (int i = 0; i < k; i++) {
    if (pipe(fd[i])) {
      perror("pipe");
      exit(1);
    }
    poll_fds[i].fd = fd[i][0];
    poll_fds[i].events = POLLIN;

    /* Evenly divide the remaining workers among the remaining children. */
    int n = (numWorkers + ((k - i) / 2)) / (k - i);
    numWorkers -= n;

    if (fork() == 0) {
      close(fd[i][0]);

      if (doWork) {
        /* Leaf: pin (optional), process our range, signal the parent. */
        if (PIN_CPU) {
          cpu_set_t cpu_set;
          CPU_ZERO(&cpu_set);
          CPU_SET(workers[new_id].cpuId, &cpu_set);
          if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set) == -1) {
            perror("sched_setaffinity");
          }
        }

        start_worker(workers + new_id, childResults[i]);
        if (write(fd[i][1], "0", 1) < 0) {
          perror("write");
          exit(1);
        }
        exit(0);
      }

      /* Interior: recurse with our slice of n workers. */
      process(new_id, workers, n, fd[i][1], childResults[i]);
      if (UNMAP) {
        while (wait(NULL) != -1) {}
      }
      exit(0);
    }
    new_id += n;
  }

  /* Merge children as they finish reporting over their pipes. */
  int childrenFinished = 0;
  while (childrenFinished < k) {
    poll(poll_fds, k, -1);
    for (int i = 0; i < k; i++) {
      if (poll_fds[i].revents & POLLIN) {
        char buffer[4];
        ssize_t num_bytes = read(poll_fds[i].fd, buffer, sizeof(buffer));
        if (num_bytes > 0) {
          childrenFinished++;

          TIMER_RESET();
          merge(out, childResults[i]);
          TIMER_MS("merge");
        }
      }
    }
  }

  /* Interior node: tell our own parent we're done. */
  if (fdOut != -1) {
    if (write(fdOut, "0", 1) < 0) {
      perror("parrent write");
      exit(1);
    }
    if (UNMAP) {
      while (wait(NULL) != -1) {}
    }
    exit(0);
  }

  if (UNMAP) {
    TIMER_RESET();
    while (wait(NULL) != -1) {}
    TIMER_MS("unmap");
  }
}

/* ========================================================================== *
 *  Leaf worker: map the file range and parse it chunk by chunk
 * ========================================================================== */

void start_worker(Worker *w, Results *out) {
  TIMER_INIT();

  /* One arena for all hash structures. The offsets table wants 4 KiB pages; the
   * rest wants 2 MiB huge pages, so we over-allocate by one huge page and align
   * the huge-page region. Everything except the long-city table is pre-faulted. */
  void *hashData = mmap(NULL, HASH_MEMORY_SIZE + HUGE_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  hashData += PACKED_OFFSETS_SIZE;
  hashData = HUGE_PAGE_CEIL_P(hashData);
  hashData -= PACKED_OFFSETS_SIZE;

  madvise(hashData + PACKED_OFFSETS_SIZE, HASHED_CITIES_SIZE + HASHED_DATA_SIZE + HASHED_CITIES_LONG_SIZE, MADV_HUGEPAGE);
  madvise(hashData, HASH_MEMORY_SIZE, MADV_POPULATE_WRITE);
  TIMER_MS_NUM("mmap", w->workerId);

  int * packedOffsets = hashData;
  hashData += PACKED_OFFSETS_SIZE;

  void * hashedCities = hashData;
  hashData += HASHED_CITIES_SIZE;

  void * hashedStorage = hashData;
  hashData += HASHED_DATA_SIZE;

  void * hashedCitiesLong = hashData;
  hashData += HASHED_CITIES_LONG_SIZE;

  Hash hash = {{packedOffsets, hashedCities, hashedStorage, hashedCitiesLong}, {0, 0}};

  /* Reusable 4 GiB virtual window. The file is mapped start after the DUMMY_PAGE.*/
  void * const data = mmap(NULL, MMAP_DATA_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  /* Fill the dummy region with copies of the dummy line "\0AD;0.0\n". */
  __m256i dummyData = _mm256_set1_epi64x(0x0A302E303B444100);
  for (int i = 0; i < DUMMY_SIZE; i += 32) {
    _mm256_store_si256(data + i, dummyData);
  }

  /* Walk the worker's range in <=MAX_CHUNK_SIZE pieces, mmap()ing a chunk of the input for each. */
  for (long start = w->start; start < w->end; start += MAX_CHUNK_SIZE) {
    long end = start + MAX_CHUNK_SIZE > w->end ? w->end : start + MAX_CHUNK_SIZE;

    bool first = w->first && start == w->start;
    bool last = w->last && end == w->end;

    uint32_t chunk_size = (uint32_t)(end - start);
    /* For non-final chunks, map one extra page so a line straddling the end is
     * fully readable; the final chunk only needs to page-align its length. */
    uint32_t mapped_file_length = last ? PAGE_CEIL(chunk_size) : chunk_size + PAGE_SIZE;

    mmap(data + DUMMY_SIZE, mapped_file_length, PROT_READ, MAP_PRIVATE | MAP_FIXED, w->fd, start);

    if (DEBUG && w->warmup) {
      long dummy = 0;
      for (long i = 0; i < mapped_file_length; i += PAGE_SIZE) {
        dummy += *(long *)(data + i);
      }
      volatile long dummy2 = dummy;
      (void)dummy2;
      TIMER_MS_NUM("warmup", w->workerId);
      TIMER_RESET();
    }

    /* Split this chunk into STRIDE sub-ranges aligned to line starts. Lane i
     * begins at the first line at-or-after chunk_size/STRIDE * i, and ends where
     * lane i+1 begins. offsets[] are absolute offsets into 'data'. */
    uint32_t offsets[STRIDE + 1];
    if (first) {
      offsets[0] = DUMMY_SIZE; /* very first byte of the file: no preceding newline */
    }
    for (int i = first ? 1 : 0; i < STRIDE; i++) {
      offsets[i] = find_next_row(data, chunk_size / STRIDE * i + DUMMY_SIZE);
    }
    offsets[STRIDE] = last ? chunk_size + DUMMY_SIZE : find_next_row(data, chunk_size + DUMMY_SIZE);

    process_chunk(data, offsets, &hash);
    TIMER_MS_NUM("chunk", w->workerId);
  }

  TIMER_RESET();
  convert_hash_to_results(&hash, out);
  TIMER_MS_NUM("convert", w->workerId);
}

/* ========================================================================== *
 *  process_chunk — the main logic
 *
 *  Parses STRIDE(=8) lines in parallel, one per AVX2 lane, until every lane has
 *  reached the end of its sub-range. Each iteration: find the ';' in each line,
 *  mask out the city name, hash + insert all 8 cities, parse all 8 temperatures,
 *  and add temperature data into each city's per-lane accumulator.
 *
 *  Input format reminder: "<city>;<temp>\n" where <temp> is X.X / XX.X / -X.X /
 *  -XX.X. A "short" city is one whose ';' lies within the first 32 bytes.
 * ========================================================================== */

__attribute__((aligned(4096))) void process_chunk(const void * const restrict base, const uint32_t *offsets, Hash * restrict hashOut) {
  /* Work on a local copy of the hash counts/pointers; flush back on exit. */
  alignas(64) Hash hash = *hashOut;
  alignas(64) uint64_t nums[STRIDE];
  alignas(32) uint32_t starts[STRIDE];      /* current line-start offset per lane */
  alignas(32) uint32_t cityHashes[STRIDE];
  bool checkFinished;

  /* starts_v: where each lane is currently parsing.
   * ends_v:   where each lane must stop (== the next lane's start). */
  __m256i starts_v = _mm256_loadu_si256((__m256i *)offsets);
  __m256i ends_v = _mm256_loadu_si256((__m256i *)(offsets + 1));
  __m256i finished_v = _mm256_set1_epi32(0); /* sticky per-lane "done" mask */

  /* A lane is done when its cursor reaches its end. */
  __m256i atEndMask = _mm256_cmpeq_epi32(starts_v, ends_v);
  checkFinished = !_mm256_testz_si256(atEndMask, atEndMask);

  _mm256_store_si256((__m256i *)starts, starts_v);

  /* Pre-occupy the dummy city's slot so finished lanes (which parse the dummy
   * line) match it instead of allocating a real city entry. */
  __m256i dummy = _mm256_load_si256(MASKED_DUMMY);
  _mm256_store_si256(hash.p.hashedCities + hash_city(dummy), dummy);

  while (1) {
    /* --- Handle lanes that have reached their end -------------------------
     * Once a lane is finished we keep it "parsing" the dummy line (start := 0,
     * which points at the dummy region) so the SIMD lanes stay full. The
     * finished_v mask is sticky; when all 8 lanes are finished we flush and
     * return. */
    if (unlikely(checkFinished)) {
      finished_v = _mm256_or_si256(finished_v, atEndMask);

      if (unlikely(_mm256_movemask_epi8(finished_v) == 0xFFFFFFFF)) {
        hashOut->counts = hash.counts;
        return;
      }

      /* Finished lanes: set start to 0 (start of dummy region), end -> DUMMY_SIZE (end of dummy region) */
      starts_v = _mm256_andnot_si256(finished_v, starts_v);
      ends_v = (__m256i)_mm256_blendv_ps((__m256)ends_v, (__m256)_mm256_set1_epi32(DUMMY_SIZE), (__m256)finished_v);


      /* We should be able to do an unconditional store,
	   * but it causes Clang to reorganize the code terribly (~10% slowdown). */
      //_mm256_store_si256((__m256i *)starts, starts_v);
      _mm256_maskstore_epi32((int *)starts, finished_v, starts_v);
    }

    /* --- Load 32 bytes from each line start (enough for any short city) ---- */
    __m256i rawCity0 = _mm256_loadu_si256(base + starts[0]);
    __m256i rawCity1 = _mm256_loadu_si256(base + starts[1]);
    __m256i rawCity2 = _mm256_loadu_si256(base + starts[2]);
    __m256i rawCity3 = _mm256_loadu_si256(base + starts[3]);
    __m256i rawCity4 = _mm256_loadu_si256(base + starts[4]);
    __m256i rawCity5 = _mm256_loadu_si256(base + starts[5]);
    __m256i rawCity6 = _mm256_loadu_si256(base + starts[6]);
    __m256i rawCity7 = _mm256_loadu_si256(base + starts[7]);

    /* --- Find the ';' in each line. semicolonBytesN is its byte offset from
     * the line start, or 32 if there is no ';' in the first 32 bytes (=> a long
     * city, handled below). --------------------------------------------------*/
    __m256i semicolons = _mm256_set1_epi8(';');
    int semicolonBytes0 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity0, semicolons)));
    int semicolonBytes1 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity1, semicolons)));
    int semicolonBytes2 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity2, semicolons)));
    int semicolonBytes3 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity3, semicolons)));
    int semicolonBytes4 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity4, semicolons)));
    int semicolonBytes5 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity5, semicolons)));
    int semicolonBytes6 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity6, semicolons)));
    int semicolonBytes7 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity7, semicolons)));

    /* Prefetch the next line for each lane (current line end + a bit). 127 keeps
     * the displacement byte-sized for a smaller opcode. */
    _mm_prefetch(base + starts[0] + semicolonBytes0 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[1] + semicolonBytes1 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[2] + semicolonBytes2 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[3] + semicolonBytes3 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[4] + semicolonBytes4 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[5] + semicolonBytes5 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[6] + semicolonBytes6 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[7] + semicolonBytes7 + 127, _MM_HINT_NTA);

    /* --- Mask each city name: keep the bytes before ';', zero the rest. ----
     * (CITY_MASK + 32 - n) gives n leading 0xFF bytes. */
    __m256i rawMask0 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes0);
    __m256i rawMask1 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes1);
    __m256i rawMask2 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes2);
    __m256i rawMask3 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes3);
    __m256i rawMask4 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes4);
    __m256i rawMask5 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes5);
    __m256i rawMask6 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes6);
    __m256i rawMask7 = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes7);

    __m256i maskedCity0 = _mm256_and_si256(rawCity0, rawMask0);
    __m256i maskedCity1 = _mm256_and_si256(rawCity1, rawMask1);
    __m256i maskedCity2 = _mm256_and_si256(rawCity2, rawMask2);
    __m256i maskedCity3 = _mm256_and_si256(rawCity3, rawMask3);
    __m256i maskedCity4 = _mm256_and_si256(rawCity4, rawMask4);
    __m256i maskedCity5 = _mm256_and_si256(rawCity5, rawMask5);
    __m256i maskedCity6 = _mm256_and_si256(rawCity6, rawMask6);
    __m256i maskedCity7 = _mm256_and_si256(rawCity7, rawMask7);

    /* --- Long-city fallback ------------------------------------------------
     * A semicolon byte count of 32 means the name didn't end in the first 32
     * bytes. process_long() handles the full (up to 128-byte) name, returns the
     * replacement packed city (a long-city reference) and the true ';' offset. */
    __m256i semicolons_v = _mm256_set_epi32(semicolonBytes7, semicolonBytes6, semicolonBytes5, semicolonBytes4, semicolonBytes3, semicolonBytes2, semicolonBytes1, semicolonBytes0);
    __m256i longCities = _mm256_cmpeq_epi32(semicolons_v, _mm256_set1_epi32(32));

    if (unlikely(!_mm256_testz_si256(longCities, longCities))) {
      if (semicolonBytes0 == 32) {
        maskedCity0 = process_long(base + starts[0], &hash, &semicolonBytes0);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes0, 0);
      }
      if (semicolonBytes1 == 32) {
        maskedCity1 = process_long(base + starts[1], &hash, &semicolonBytes1);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes1, 1);
      }
      if (semicolonBytes2 == 32) {
        maskedCity2 = process_long(base + starts[2], &hash, &semicolonBytes2);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes2, 2);
      }
      if (semicolonBytes3 == 32) {
        maskedCity3 = process_long(base + starts[3], &hash, &semicolonBytes3);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes3, 3);
      }
      if (semicolonBytes4 == 32) {
        maskedCity4 = process_long(base + starts[4], &hash, &semicolonBytes4);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes4, 4);
      }
      if (semicolonBytes5 == 32) {
        maskedCity5 = process_long(base + starts[5], &hash, &semicolonBytes5);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes5, 5);
      }
      if (semicolonBytes6 == 32) {
        maskedCity6 = process_long(base + starts[6], &hash, &semicolonBytes6);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes6, 6);
      }
      if (semicolonBytes7 == 32) {
        maskedCity7 = process_long(base + starts[7], &hash, &semicolonBytes7);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes7, 7);
      }
    }

    /* --- Hash all 8 cities at once ----------------------------------------
     * hash_cities() returns the 8 hashes in the permuted lane order
     * [a, c, e, g, b, d, f, h]; the cityHashes[] indices used below are
     * permuted to match (see the insert_city calls). */
    __m256i city_hashes = hash_cities(maskedCity0, maskedCity1, maskedCity2, maskedCity3, maskedCity4, maskedCity5, maskedCity6, maskedCity7);
    _mm256_store_si256((__m256i *)cityHashes, city_hashes);

    /* Advance each cursor past the city name to the ';'. */
    starts_v = _mm256_add_epi32(starts_v, semicolons_v);

    /* ====================================================================== *
     *  Temperature parsing (8 lanes in parallel, branchless)
     *
     *  For each lane we load 8 bytes starting 2 bytes *before* the ';':
     *    byte 0,1 : last two city chars (ignored)
     *    byte 2   : ';'
     *    byte 3   : first temperature char ('-' or a digit)
     *    byte 4-7 : remaining temperature chars + '\n'
     *
     *  The 8-byte loads are gathered in a deliberately permuted order
     *  (lanes 0,1,4,5 then 2,3,6,7) so that the subsequent shuffle_ps /
     *  unpack steps land each value in the right place. nums_low / nums_high
     *  hold those two halves.
     * ====================================================================== */
    nums[0] = *(long *)(base + starts[0] + semicolonBytes0 - 2);
    nums[1] = *(long *)(base + starts[1] + semicolonBytes1 - 2);
    nums[2] = *(long *)(base + starts[4] + semicolonBytes4 - 2);
    nums[3] = *(long *)(base + starts[5] + semicolonBytes5 - 2);
    nums[4] = *(long *)(base + starts[2] + semicolonBytes2 - 2);
    nums[5] = *(long *)(base + starts[3] + semicolonBytes3 - 2);
    nums[6] = *(long *)(base + starts[6] + semicolonBytes6 - 2);
    nums[7] = *(long *)(base + starts[7] + semicolonBytes7 - 2);

    __m256i nums_low = _mm256_load_si256((__m256i *)nums);        /* lanes 0,1,4,5 */
    __m256i nums_high = _mm256_load_si256((__m256i *)(nums + 4)); /* lanes 2,3,6,7 */

    /* Split every lane's 8 bytes into its low 4 bytes ([city,city,';',c0]) and
     * high 4 bytes ([c1,c2,c3,c4]). */
    __m256i low_words = (__m256i) _mm256_shuffle_ps((__m256)nums_low, (__m256)nums_high, 0x88);
    __m256i high_words = (__m256i) _mm256_shuffle_ps((__m256)nums_low, (__m256)nums_high, 0xDD);

    /* Detect the sign. The 16-bit pattern at bytes [2,3] is [';', firstchar].
     * Comparing against (';' | '-'<<8) per byte yields: byte 2 == 0xFF always
     * (it really is ';'), byte 3 == 0xFF iff the number is negative. The always
     * -set byte 2 keeps the mask element nonzero+positive in the positive case,
     * which matters for the _mm256_sign_epi32 below. */
    __m256i minus_mask = _mm256_cmpeq_epi8(low_words, _mm256_set1_epi16(';' + ('-' << 8)));

    /* Build a version of the high word shifted so the *first digit* sits in
     * byte 0 even for positive numbers (whose first digit is at byte 3). */
    __m256i nums_low_left1 = _mm256_slli_epi64(nums_low, 8);
    __m256i nums_high_left1 = _mm256_slli_epi64(nums_high, 8);
    __m256i high_words_left1 = (__m256i) _mm256_shuffle_ps((__m256)nums_low_left1, (__m256)nums_high_left1, 0xDD);

    /* Choose the digit-aligned window:
     *   positive  -> high_words_left1 (digits start at original byte 3)
     *   negative  -> high_words       (digits start at original byte 4, skipping '-')
     * Result is left-aligned so the decimal point is in byte 1 (X.X) or byte 2
     * (XX.X). */
    __m256i nums_blended = (__m256i)_mm256_blendv_ps((__m256)high_words_left1, (__m256)high_words, (__m256)minus_mask);
    /* (~2 cycle stall on the blend result.) */

    /* Default cursor advance from the ';': ";XX.X\n" == 6 bytes. Corrected for
     * sign / short form below. */
    starts_v = _mm256_add_epi32(starts_v, _mm256_set1_epi32(6));

    /* +1 byte of line length for a leading '-'. */
    __m256i minus_mask_shift = _mm256_srli_epi32(minus_mask, 31);

    /* Detect the short form "X.X": after alignment its byte 3 is '\n'. */
    __m256i newline_mask = _mm256_cmpeq_epi8(nums_blended, _mm256_set1_epi8('\n'));

    /* -1 byte of line length for the short "X.X"/"-X.X" form. */
    __m256i newline_mask_shift = _mm256_srli_epi32(newline_mask, 31);

    /* For the short form, shift left one more byte so the decimal point always
     * lands in byte 2 — unifying the layout to [tens, ones, '.', tenths]
     * (tens == 0 for the short form). */
    __m256i newline_shift = _mm256_slli_epi32(newline_mask_shift, 3);
    nums_blended = _mm256_sllv_epi32(nums_blended, newline_shift);

    /* ASCII -> digit. Subtract '0' with unsigned saturation so the '.' (0x2E,
     * below '0') and any absent tens digit both saturate to 0. Now each lane
     * holds bytes [tens, ones, 0, tenths]. */
    __m256i numbers = _mm256_subs_epu8(nums_blended, _mm256_set1_epi8('0'));

    /* Horizontal decimal combine:
     *   maddubs with [100,10,0,1] -> 16-bit pairs [tens*100+ones*10, tenths]
     *   madd    with [1,1]        -> 32-bit  tens*100 + ones*10 + tenths
     * i.e. the magnitude in tenths of a degree. */
    __m256i mulled = _mm256_maddubs_epi16(numbers, _mm256_set1_epi32(0x01000a64));
    mulled = _mm256_madd_epi16(mulled, _mm256_set1_epi32(0x00010001));

    /* Finalize the cursor: +1 for '-', -1 for the short form. */
    starts_v = _mm256_add_epi32(starts_v, minus_mask_shift);
    starts_v = _mm256_sub_epi32(starts_v, newline_mask_shift);
    _mm256_store_si256((__m256i *)(starts), starts_v);

    /* Re-check which lanes have now reached their end. */
    atEndMask = _mm256_cmpeq_epi32(starts_v, ends_v);
    checkFinished = !_mm256_testz_si256(atEndMask, atEndMask);

    /* Apply the sign: sign_epi32 negates 'mulled' where minus_mask is negative
     * (byte 3 set), keeps it where positive. 'final' is the signed temperature
     * in tenths, per lane. */
    __m256i final = _mm256_sign_epi32(mulled, minus_mask);

    /* ====================================================================== *
     *  Insert each city and fold its temperature into the accumulators.
     *
     *  insert_city() returns the slot's byte offset (in the short-city table).
     *  The matching accumulator block lives at hashedStorage + 4*offset, and
     *  each lane uses its own 16-byte HashEntry within it (+ 16*lane).
     *
     *  Interleaved load/insert order keeps the 8 dependent probes in flight.
     *  cityHashes[] is indexed in hash_cities()'s permuted order (a,c,e,g,b,d,f,h).
     * ====================================================================== */
    long hash0 = insert_city(&hash, cityHashes[0], maskedCity0);
    __m128i vals0 = _mm_load_si128(hash.p.hashedStorage + 4*hash0 + 16*0);

    long hash4 = insert_city(&hash, cityHashes[2], maskedCity4);
    __m128i vals4 = _mm_load_si128(hash.p.hashedStorage + 4*hash4 + 16*4);

    long hash1 = insert_city(&hash, cityHashes[4], maskedCity1);
    __m128i vals1 = _mm_load_si128(hash.p.hashedStorage + 4*hash1 + 16*1);

    long hash5 = insert_city(&hash, cityHashes[6], maskedCity5);
    __m128i vals5 = _mm_load_si128(hash.p.hashedStorage + 4*hash5 + 16*5);

    long hash2 = insert_city(&hash, cityHashes[1], maskedCity2);
    __m128i vals2 = _mm_load_si128(hash.p.hashedStorage + 4*hash2 + 16*2);

    long hash6 = insert_city(&hash, cityHashes[3], maskedCity6);
    __m128i vals6 = _mm_load_si128(hash.p.hashedStorage + 4*hash6 + 16*6);

    long hash3 = insert_city(&hash, cityHashes[5], maskedCity3);
    __m128i vals3 = _mm_load_si128(hash.p.hashedStorage + 4*hash3 + 16*3);

    long hash7 = insert_city(&hash, cityHashes[7], maskedCity7);
    __m128i vals7 = _mm_load_si128(hash.p.hashedStorage + 4*hash7 + 16*7);

    /* Pair up lanes into 256-bit registers: (lane0,lane4), (1,5), (2,6), (3,7). */
    __m256i ae = _mm256_set_m128i(vals4, vals0);
    __m256i bf = _mm256_set_m128i(vals5, vals1);
    __m256i cg = _mm256_set_m128i(vals6, vals2);
    __m256i dh = _mm256_set_m128i(vals7, vals3);

    /* Separate the packed sum/count (low 64 bits) from min/negmax (high 64). */
    __m256i abef_low = _mm256_unpacklo_epi64(ae, bf);
    __m256i cdgh_low = _mm256_unpacklo_epi64(cg, dh);

    __m256i abef_high = _mm256_unpackhi_epi64(ae, bf);
    __m256i cdgh_high = _mm256_unpackhi_epi64(cg, dh);

    /* Sign-extend each lane's 32-bit 'final' to 64 bits, positioned to add into
     * the corresponding packed sum/count word. */
    __m256i abef_shift = _mm256_set_epi64x(0x0707070707060504, 0x0303030303020100, 0x0707070707060504, 0x0303030303020100);
    __m256i final_abef = _mm256_shuffle_epi8(final, abef_shift);
    __m256i cdgh_shift = _mm256_set_epi64x(0x0F0F0F0F0F0E0D0C, 0x0B0B0B0B0B0A0908, 0x0F0F0F0F0F0E0D0C, 0x0B0B0B0B0B0A0908);
    __m256i final_cdgh = _mm256_shuffle_epi8(final, cdgh_shift);

    /* count += 1 lives in the high bits of the packed word. */
    __m256i inc = _mm256_set1_epi64x(1L << COUNT_BITS_START);

    __m256i new_abef_low = _mm256_add_epi64(abef_low, final_abef);
    new_abef_low = _mm256_add_epi64(new_abef_low, inc);

    __m256i new_cdgh_low = _mm256_add_epi64(cdgh_low, final_cdgh);
    new_cdgh_low = _mm256_add_epi64(new_cdgh_low, inc);

    /* min/max update. We store (min, -max); interleaving (final, -final) lets a
     * single signed min handle both: min(min, x) and min(-max, -x) == -max(max, x). */
    __m256i negfinal = _mm256_sub_epi32(_mm256_setzero_si256(), final);
    __m256i abef_minmax = _mm256_unpacklo_epi32(final, negfinal);
    __m256i cdgh_minmax = _mm256_unpackhi_epi32(final, negfinal);

    __m256i new_abef_high = _mm256_min_epi32(abef_minmax, abef_high);
    __m256i new_cdgh_high = _mm256_min_epi32(cdgh_minmax, cdgh_high);

    /* Re-pack low (sum/count) and high (min/negmax) halves back per lane. */
    __m256i new_ae = _mm256_unpacklo_epi64(new_abef_low, new_abef_high);
    __m256i new_bf = _mm256_unpackhi_epi64(new_abef_low, new_abef_high);
    __m256i new_cg = _mm256_unpacklo_epi64(new_cdgh_low, new_cdgh_high);
    __m256i new_dh = _mm256_unpackhi_epi64(new_cdgh_low, new_cdgh_high);

    /* Store each lane's updated HashEntry back to its slot. */
    _mm_store_si128(hash.p.hashedStorage + 4*hash0 + 16*0, _mm256_extracti128_si256(new_ae, 0));
    _mm_store_si128(hash.p.hashedStorage + 4*hash1 + 16*1, _mm256_extracti128_si256(new_bf, 0));
    _mm_store_si128(hash.p.hashedStorage + 4*hash2 + 16*2, _mm256_extracti128_si256(new_cg, 0));
    _mm_store_si128(hash.p.hashedStorage + 4*hash3 + 16*3, _mm256_extracti128_si256(new_dh, 0));
    _mm_store_si128(hash.p.hashedStorage + 4*hash4 + 16*4, _mm256_extracti128_si256(new_ae, 1));
    _mm_store_si128(hash.p.hashedStorage + 4*hash5 + 16*5, _mm256_extracti128_si256(new_bf, 1));
    _mm_store_si128(hash.p.hashedStorage + 4*hash6 + 16*6, _mm256_extracti128_si256(new_cg, 1));
    _mm_store_si128(hash.p.hashedStorage + 4*hash7 + 16*7, _mm256_extracti128_si256(new_dh, 1));
  }
}

/* ========================================================================== *
 *  Long-city handling
 * ========================================================================== */

/*
 * Hash the first 16 bytes of a long city name as quickly as possible.
 */
int hash_long(long x, long y) {
  long seed = 0x9e3779b97f4a7c15; /* ~fxhash */
  return ((_lrotl(x * seed, 5) ^ y) * seed) & HASH_LONG_MASK;
}

/*
 * Parse a city name longer than 31 bytes. Loads up to four 32-byte segments,
 * finds the ';' in segments 1..3 (segment 0 is known to have none), inserts the
 * full name into the long-city table, writes the true ';' offset to
 * *semicolonBytesOut, and returns the inline packed-city reference to store.
 */
__m256i process_long(const void * const restrict start, Hash * restrict h, int * restrict semicolonBytesOut) {
  __m256i seg0 = _mm256_loadu_si256(start +  0);
  __m256i seg1 = _mm256_loadu_si256(start + 32);
  __m256i seg2 = _mm256_loadu_si256(start + 64);
  __m256i seg3 = _mm256_loadu_si256(start + 96);

  __m256i semicolons = _mm256_set1_epi8(';');
  int semicolonBytes1 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg1, semicolons)));
  int semicolonBytes2 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg2, semicolons)));
  int semicolonBytes3 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg3, semicolons)));

  /* only hash the first 16 bytes, longer doesn't have any benefits */
  int hash = hash_long(*(long *)start, *((long *)start + 1));

  /* The ';' is in segment 1, 2, or 3. Mask the final segment, then insert with
   * the matching arity. *semicolonBytesOut is the absolute offset from start. */
  if (semicolonBytes1 < 32) {
    *semicolonBytesOut = 32 + semicolonBytes1;
    __m256i mask = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes1);
    seg1 = _mm256_and_si256(seg1, mask);
    hash = insert_city_long1(h, hash, seg0, seg1);
  }
  else if (semicolonBytes2 < 32) {
    *semicolonBytesOut = 64 + semicolonBytes2;
    __m256i mask = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes2);
    seg2 = _mm256_and_si256(seg2, mask);
    hash = insert_city_long2(h, hash, seg0, seg1, seg2);
  }
  else {
    *semicolonBytesOut = 96 + semicolonBytes3;
    __m256i mask = _mm256_loadu_si256(CITY_MASK + 32 - semicolonBytes3);
    seg3 = _mm256_and_si256(seg3, mask);
    hash = insert_city_long3(h, hash, seg0, seg1, seg2, seg3);
  }

  return city_from_long_hash(hash);
}

/* ========================================================================== *
 *  Hashing
 * ========================================================================== */

/*
 * Hash 8 cities at once. Only the first 8 bytes of each name feed the hash (with
 * a high-nibble fold mixed in); collisions are resolved by full-name comparison
 * during linear probing, so this only affects probe length, not correctness.
 *
 * The 8 results come back in the permuted lane order [a, c, e, g, b, d, f, h].
 */
__attribute__((always_inline)) inline __m256i hash_cities(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {
  __m256i ab = _mm256_inserti128_si256(a, _mm256_castsi256_si128(b), 1);
  __m256i cd = _mm256_inserti128_si256(c, _mm256_castsi256_si128(d), 1);
  __m256i ef = _mm256_inserti128_si256(e, _mm256_castsi256_si128(f), 1);
  __m256i gh = _mm256_inserti128_si256(g, _mm256_castsi256_si128(h), 1);

  cd = _mm256_slli_si256(cd, 8);
  gh = _mm256_slli_si256(gh, 8);

  __m256i acbd = _mm256_blend_epi32(ab, cd, 0xCC);
  __m256i egfh = _mm256_blend_epi32(ef, gh, 0xCC);

  /* fold the high nibbles in so bytes beyond 0..3 still affect the hash */
  __m256i acbd2 = _mm256_srli_epi64(acbd, 28);
  __m256i egfh2 = _mm256_srli_epi64(egfh, 28);

  acbd = _mm256_xor_si256(acbd, acbd2);
  egfh = _mm256_xor_si256(egfh, egfh2);

  __m256i acegbdfh = (__m256i) _mm256_shuffle_ps((__m256)acbd, (__m256)egfh, 0x88);

  __m256i hash = _mm256_madd_epi16(acegbdfh, acegbdfh);
  __m256i hash_mask = _mm256_set1_epi32(HASH_SHORT_MASK);
  return _mm256_and_si256(hash, hash_mask);
}

/* Single-city convenience wrapper used by the dummy slot, merge(), etc. */
__attribute__((always_inline)) inline int hash_city(__m256i str) {
  __m256i zero = _mm256_set1_epi32(0);
  __m256i hash = hash_cities(str, zero, zero, zero, zero, zero, zero, zero);
  return _mm256_extract_epi32(hash, 0);
}

/* ========================================================================== *
 *  Insertion (open addressing, linear probing)
 * ========================================================================== */

/*
 * Find or insert a short city, returning its slot's byte offset. On a fresh
 * insert: store the name, record the offset for later enumeration, and
 * initialize all STRIDE accumulators (sum biased to 0, count 0, min=MAX_TEMP,
 * negmax=-MIN_TEMP).
 */
__attribute__((always_inline)) inline long insert_city(Hash * restrict h, long hash, const __m256i maskedCity) {
  while (1) {
    __m256i stored = _mm256_load_si256(h->p.hashedCities + hash);
    __m256i xor = _mm256_xor_si256(maskedCity, stored);
    if (likely(_mm256_testz_si256(xor, xor))) {
      return hash; /* exact match */
    }
    if (_mm256_testz_si256(stored, stored)) {
      /* Empty slot: claim it. */
      _mm256_store_si256(h->p.hashedCities + hash, maskedCity);
      h->p.packedOffsets[h->counts.numCities] = hash;
      h->counts.numCities += 1;

      __m256i initData = _mm256_set_epi32(-MIN_TEMP, MAX_TEMP, SUM_SIGN_BIT >> 32, 0,
                                          -MIN_TEMP, MAX_TEMP, SUM_SIGN_BIT >> 32, 0);
      _mm256_store_si256(h->p.hashedStorage + 4*hash +  0, initData);
      _mm256_store_si256(h->p.hashedStorage + 4*hash + 32, initData);
      _mm256_store_si256(h->p.hashedStorage + 4*hash + 64, initData);
      _mm256_store_si256(h->p.hashedStorage + 4*hash + 96, initData);
      return hash;
    }
    hash += SHORT_CITY_LENGTH;
    hash &= HASH_SHORT_MASK;
  }
}

/* Long-city insert variants: 2, 3, or 4 segments of the name are significant
 * (the rest is zeroed by the caller's mask). Each probes the long-city table by
 * full-name comparison and inserts on the first empty slot. */
int insert_city_long1(Hash * restrict hash, int hash_value, __m256i seg0, __m256i seg1) {
  while (1) {
    __m256i stored0 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value +  0);
    __m256i stored1 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 32);
    __m256i xor0 = _mm256_xor_si256(stored0, seg0);
    __m256i xor1 = _mm256_xor_si256(stored1, seg1);

    if (_mm256_testz_si256(xor0, xor0) && _mm256_testz_si256(xor1, xor1)) {
      return hash_value;
    }

    if (_mm256_testz_si256(stored0, stored0)) {
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value +  0, seg0);
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value + 32, seg1);
      hash->counts.numCitiesLong++;
      return hash_value;
    }
    hash_value += LONG_CITY_LENGTH;
    hash_value &= HASH_LONG_MASK;
  }
}

int insert_city_long2(Hash * restrict hash, int hash_value, __m256i seg0, __m256i seg1, __m256i seg2) {
  while (1) {
    __m256i stored0 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value +  0);
    __m256i stored1 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 32);
    __m256i stored2 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 64);
    __m256i xor0 = _mm256_xor_si256(stored0, seg0);
    __m256i xor1 = _mm256_xor_si256(stored1, seg1);
    __m256i xor2 = _mm256_xor_si256(stored2, seg2);

    if (_mm256_testz_si256(xor0, xor0) && _mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2)) {
      return hash_value;
    }

    if (_mm256_testz_si256(stored0, stored0)) {
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value +  0, seg0);
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value + 32, seg1);
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value + 64, seg2);
      hash->counts.numCitiesLong++;
      return hash_value;
    }
    hash_value += LONG_CITY_LENGTH;
    hash_value &= HASH_LONG_MASK;
  }
}

int insert_city_long3(Hash * restrict hash, int hash_value, __m256i seg0, __m256i seg1, __m256i seg2, __m256i seg3) {
  while (1) {
    __m256i stored0 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value +  0);
    __m256i stored1 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 32);
    __m256i stored2 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 64);
    __m256i stored3 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 96);
    __m256i xor0 = _mm256_xor_si256(stored0, seg0);
    __m256i xor1 = _mm256_xor_si256(stored1, seg1);
    __m256i xor2 = _mm256_xor_si256(stored2, seg2);
    __m256i xor3 = _mm256_xor_si256(stored3, seg3);

    if (_mm256_testz_si256(xor0, xor0) && _mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2) && _mm256_testz_si256(xor3, xor3)) {
      return hash_value;
    }

    if (_mm256_testz_si256(stored0, stored0)) {
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value +  0, seg0);
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value + 32, seg1);
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value + 64, seg2);
      _mm256_store_si256(hash->p.hashedCitiesLong + hash_value + 96, seg3);
      hash->counts.numCitiesLong++;
      return hash_value;
    }
    hash_value += LONG_CITY_LENGTH;
    hash_value &= HASH_LONG_MASK;
  }
}

/* ========================================================================== *
 *  Hash -> Results, and merging Results across workers
 * ========================================================================== */

/*
 * Collapse a worker's raw hash table into a Results table: for each occupied
 * slot, sum the STRIDE per-lane accumulators into one {sum, count, min, max},
 * copy any long-city name into the Results' own long-city array, and insert
 * into the results table.
 */
void convert_hash_to_results(Hash * restrict hash, Results * restrict out) {
  out->numCities = hash->counts.numCities;
  out->numLongCities = 0;

  for (int i = 0; i < hash->counts.numCities; i++) {
    int offset = hash->p.packedOffsets[i];
    PackedCity city = { .reg = _mm256_load_si256(hash->p.hashedCities + offset)};
    HashEntry *entries = hash->p.hashedStorage + offset * (HASH_ENTRY_SIZE / SHORT_CITY_LENGTH);

    /* Reduce the STRIDE lane accumulators. */
    long sum   = EXTRACT_SUM(entries[0].packedSumCount);
    int count  = EXTRACT_COUNT(entries[0].packedSumCount);
    int min    = entries[0].min;
    int negmax = entries[0].negmax;

    for (int j = 1; j < STRIDE; j++) {
      sum += EXTRACT_SUM(entries[j].packedSumCount);
      count += EXTRACT_COUNT(entries[j].packedSumCount);
      min = MIN(min, entries[j].min);
      negmax = MIN(negmax, entries[j].negmax);
    }

    /* Relocate long-city names into the Results arena and rewrite the ref. */
    if (unlikely(city_is_long(city))) {
      LongCity *longCity = hash->p.hashedCitiesLong + city.longRef.index;
      out->longCities[out->numLongCities] = *longCity;

      city.longRef.index = out->numLongCities;
      out->numLongCities++;
    }

    /* Insert into the results table (linear probe on empty slot). */
    offset = (offset >> (HASH_SHIFT - HASH_RESULT_SHIFT)) & HASH_RESULT_MASK;
    while (1) {
      if (_mm256_testz_si256(out->rows[offset / SHORT_CITY_LENGTH].city.reg, out->rows[offset / SHORT_CITY_LENGTH].city.reg)) {
        out->rows[offset / SHORT_CITY_LENGTH] = (ResultsRow) {city, sum, count, min, -negmax};
        break;
      }
      offset += SHORT_CITY_LENGTH;
      offset &= HASH_RESULT_MASK;
    }
    out->refs[i] = (ResultsRef) {offset};
  }
}

/* Whole-name equality for long cities (all four 256-bit segments). */
__attribute__((always_inline)) inline bool long_city_equal(LongCity *a, LongCity *b) {
  __m256i xor0 = _mm256_xor_si256(a->regs[0], b->regs[0]);
  __m256i xor1 = _mm256_xor_si256(a->regs[1], b->regs[1]);
  __m256i xor2 = _mm256_xor_si256(a->regs[2], b->regs[2]);
  __m256i xor3 = _mm256_xor_si256(a->regs[3], b->regs[3]);
  return _mm256_testz_si256(xor0, xor0) && _mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2) && _mm256_testz_si256(xor3, xor3);
}

/*
 * Merge src's results into dst. For each src city: relocate/dedupe any long
 * name into dst, then probe dst's results table and either accumulate into the
 * existing row or claim an empty slot.
 */
void merge(Results * restrict dst, Results * restrict src) {
  for (int i = 0; i < src->numCities; i++) {
    ResultsRef ref = src->refs[i];
    ResultsRow row = src->rows[ref.offset / SHORT_CITY_LENGTH];
    int hashValue = hash_city(row.city.reg);

    if (unlikely(city_is_long(row.city))) {
      LongCity *longCity = src->longCities + row.city.longRef.index;

      /* Find this long name in dst, or append it. */
      int dstLongCityIdx = 0;
      LongCity *dstLongCity;
      for (; dstLongCityIdx < dst->numLongCities; dstLongCityIdx++) {
        dstLongCity = dst->longCities + dstLongCityIdx;
        if (long_city_equal(longCity, dstLongCity)) {
          break;
        }
      }
      if (dstLongCityIdx == dst->numLongCities) {
        dst->longCities[dst->numLongCities] = *longCity;
        dst->numLongCities++;
      }

      row.city = (PackedCity) city_from_long_hash(dstLongCityIdx);
      hashValue = hash_city(row.city.reg);
    }

    hashValue = (hashValue >> (HASH_SHIFT - HASH_RESULT_SHIFT)) & HASH_RESULT_MASK;
    while (1) {
      ResultsRow *dstRow = dst->rows + (hashValue / SHORT_CITY_LENGTH);
      __m256i xor = _mm256_xor_si256(dstRow->city.reg, row.city.reg);
      if (likely(_mm256_testz_si256(xor, xor))) {
        /* Same city already in dst: accumulate. */
        dstRow->sum += row.sum;
        dstRow->count += row.count;
        dstRow->min = MIN(dstRow->min, row.min);
        dstRow->max = MAX(dstRow->max, row.max);
        break;
      }

      if (_mm256_testz_si256(dstRow->city.reg, dstRow->city.reg)) {
        /* Empty slot: copy the row in and register it. */
        dst->refs[dst->numCities] = (ResultsRef){hashValue};
        dst->rows[hashValue / SHORT_CITY_LENGTH] = row;
        dst->numCities++;
        break;
      }

      hashValue += SHORT_CITY_LENGTH;
      hashValue &= HASH_RESULT_MASK;
    }
  }
}

/* ========================================================================== *
 *  Output
 * ========================================================================== */

/* qsort_r comparator: alphabetical by city name (short inline or long). */
int sort_result(const void *a, const void *b, void *arg) {
  Results *r = arg;
  const ResultsRef *left = a;
  const ResultsRef *right = b;

  PackedCity leftCity = r->rows[left->offset / SHORT_CITY_LENGTH].city;
  PackedCity rightCity = r->rows[right->offset / SHORT_CITY_LENGTH].city;

  char *leftBytes = leftCity.shortCity.bytes;
  char *rightBytes = rightCity.shortCity.bytes;

  if (unlikely(city_is_long(leftCity))) {
    leftBytes = r->longCities[leftCity.longRef.index].bytes;
  }
  if (unlikely(city_is_long(rightCity))) {
    rightBytes = r->longCities[rightCity.longRef.index].bytes;
  }
  return strcmp(leftBytes, rightBytes);
}

/* Emit "{city=min/mean/max, ...}\n". Stats are in tenths; mean is rounded to one
 * decimal. (One big buffer + a single fputs keeps I/O time minimal.) */
void print_results(Results *results) {
  char *buffer = malloc(MAX_CITIES * 150);

  int pos = 0;
  buffer[pos++] = '{';

  for (int i = 0; i < results->numCities; i++) {
    ResultsRef ref = results->refs[i];
    ResultsRow row = results->rows[ref.offset / SHORT_CITY_LENGTH];

    float sum = row.sum;
    float count = row.count;
    float min = row.min * 0.1;
    float max = row.max * 0.1;

    const char *bytes;
    if (unlikely(city_is_long(row.city))) {
      bytes = results->longCities[row.city.longRef.index].bytes;
    }
    else {
      bytes = row.city.shortCity.bytes;
    }
    pos += sprintf(buffer + pos, "%s=%.1f/%.1f/%.1f", bytes, min, round(sum/count) * 0.1, max);

    if (i != results->numCities - 1) {
      buffer[pos++] = ',';
      buffer[pos++] = ' ';
    }
  }
  buffer[pos++] = '}';
  buffer[pos++] = '\n';
  buffer[pos++] = '\0';
  fputs(buffer, stdout);
}

/* Return the offset of the first byte after the next '\n' at/after 'offset'. */
uint32_t find_next_row(const void *data, uint32_t offset) {
  __m256i newlines = _mm256_set1_epi8('\n');
  __m256i chars = _mm256_loadu_si256(data + offset);
  uint32_t bytes = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, newlines)));
  if (likely(bytes < 32)) {
    return offset + bytes + 1;
  }
  /* Newline not in the first 32 bytes: scan the rest. */
  while (*((char *)data + offset + bytes) != '\n') {
    bytes++;
  }
  return offset + bytes + 1;
}

/* ========================================================================== *
 *  Small helpers
 * ========================================================================== */

/* Build the inline packed city for a long city: [LONG_CITY_SENTINEL, index]. */
__attribute__((always_inline)) inline __m256i city_from_long_hash(int hashValue) {
  return _mm256_set_epi32(0, 0, 0, 0, 0, 0, hashValue, LONG_CITY_SENTINEL);
}

/* A slot holds a long-city reference iff its first word is the sentinel. */
__attribute__((always_inline)) inline bool city_is_long(PackedCity city) {
  return city.longRef.sentinel == LONG_CITY_SENTINEL;
}

/* ========================================================================== *
 *  Debug
 * ========================================================================== */

void debug_results(Results *results) {
  fprintf(stderr, "\n");
  for (int i = 0; i < MIN(10, results->numCities); i++) {
    ResultsRef ref = results->refs[i];
    ResultsRow row = results->rows[ref.offset / SHORT_CITY_LENGTH];

    const char *bytes;
    if (city_is_long(row.city)) {
      bytes = results->longCities[row.city.longRef.index].bytes;
    }
    else {
      bytes = row.city.shortCity.bytes;
    }
    fprintf(stderr, "%-100s %12ld %11d %4d %4d\n", bytes, row.sum, row.count, row.min, row.max);
  }

  long total = 0;
  for (int i = 0; i < results->numCities; i++) {
    ResultsRef ref = results->refs[i];
    ResultsRow row = results->rows[ref.offset / SHORT_CITY_LENGTH];
    total += row.count;
  }
  fprintf(stderr, "total: %ld\n", total);
}

void print256(__m256i var) {
  uint8_t val[32];
  memcpy(val, &var, sizeof(val));
  fprintf(
    stderr,
    "%02x %02x %02x %02x  %02x %02x %02x %02x | %02x %02x %02x %02x  %02x %02x %02x %02x | %02x %02x %02x %02x  %02x %02x %02x %02x | %02x %02x %02x %02x  %02x %02x %02x %02x\n",
    0xFF &  val[0], 0xFF &  val[1], 0xFF &  val[2], 0xFF &  val[3],
    0xFF &  val[4], 0xFF &  val[5], 0xFF &  val[6], 0xFF &  val[7],
    0xFF &  val[8], 0xFF &  val[9], 0xFF & val[10], 0xFF & val[11],
    0xFF & val[12], 0xFF & val[13], 0xFF & val[14], 0xFF & val[15],
    0xFF & val[16], 0xFF & val[17], 0xFF & val[18], 0xFF & val[19],
    0xFF & val[20], 0xFF & val[21], 0xFF & val[22], 0xFF & val[23],
    0xFF & val[24], 0xFF & val[25], 0xFF & val[26], 0xFF & val[27],
    0xFF & val[28], 0xFF & val[29], 0xFF & val[30], 0xFF & val[31]);
}
