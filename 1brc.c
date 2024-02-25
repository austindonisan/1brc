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

/*
 * wait() for all child processes before exiting.
 * If false, "cheat" by returning immediately, leaving orphan processes.
 */
#define UNMAP false

/*
 * Pin worker threads to a CPU.
 * Very helpfuly at high core counts, slightly harmful at low ones.
 *
 * Each worker is assigned to the lowest unused CPU # we're scheduled for.
 * For hyperthreaded CPUs, logical CPU numbers must not be interleaved.
 */
#define PIN_CPU true

/*
 * Print timing information and city summary to stderr.
 */
#define DEBUG false

#define HASH_SHIFT 17      // 17 is a happy compromise between non-10k/10k: 16 is 1% faster/10% slower; 18 is 1% slower/3% faster
#define HASH_LONG_SHIFT 14 // 14 is requried to fit 10k entries and also fastest
#define HASH_RESULT_SHIFT 14 // 14 is required to fit 10k entries and also fastest

#define MAX_CITIES 10000
#define MAX_TEMP 999
#define MIN_TEMP -999

#define SHORT_CITY_LENGTH 32
#define LONG_CITY_LENGTH 128

// wrapping and fitting nicely in pages is better than extra buffer at the end
#define HASH_ENTRIES      (1 << HASH_SHIFT)
#define HASH_LONG_ENTRIES (1 << HASH_LONG_SHIFT)

// 32 byte AVX2 registers can fit 8 values at once
// going up/down to 64/16 byte AXV512/SSE requires code changes, too
#define STRIDE 8

typedef struct {
  int64_t packedSum;
  int32_t min;
  int32_t max;
} HashEntry;

typedef struct {
  int * const restrict packedOffsets;
  void * const restrict hashedCities;
  void * const restrict hashedStorage;
  void * const restrict hashedCitiesLong;
} HashPointers;

typedef struct {
  int numCities;
  int numCitiesLong;
} HashCounts;

typedef struct {
  const HashPointers p;
  HashCounts counts;
} Hash;

typedef struct {
  long start;
  long end;
  int fd;
  int workerId;
  int cpuId;
  bool fork;
  bool warmup;
  bool first;
  bool last;
} Worker;

typedef struct {
  int64_t packedSumCount;
  int32_t min;
  int32_t max;
} HashRow;

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

typedef union {
  __m256i reg;
  ShortCity shortCity;
  LongCityRef longRef;
} PackedCity;

typedef struct {
  PackedCity city;
  int64_t sum;
  int32_t count;
  int16_t min;
  int16_t max;
} ResultsRow;

typedef struct {
  uint32_t offset;
} ResultsRef;

typedef struct {
  int numCities;
  int numLongCities;
  ResultsRef * restrict refs;
  ResultsRow * restrict rows;
  LongCity * restrict longCities;
} Results;

void prep_workers(Worker *workers, int numWorkers, bool warmup, int fd, struct stat *fileStat);
void process(int id, Worker *workers, int numWorkers, int fd, Results *out);
void start_worker(Worker *w, Results *out);
void process_chunk(const void * const restrict base, const uint32_t * offsets, Hash * restrict h);
__m256i process_long(const void * const restrict start, Hash * restrict h, int * restrict semicolonBytesOut);
inline __m256i hash_cities(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h);
inline int hash_city(__m256i str);
inline long insert_city(Hash * restrict h, long hash, const __m256i maskedCity);
int insert_city_long1(Hash * restrict h, int hash, __m256i seg0, __m256i seg1);
int insert_city_long2(Hash * restrict h, int hash, __m256i seg0, __m256i seg1, __m256i seg2);
int insert_city_long3(Hash * restrict h, int hash, __m256i seg0, __m256i seg1, __m256i seg2, __m256i seg3);
void merge(Results * restrict dst, Results * restrict src);
int sort_result(const void *a, const void *b, void *arg);
uint32_t find_next_row(const void *data, uint32_t offset);
void print_results(Results *results);
void debug_results(Results *results);
inline __m256i city_from_long_hash(int hashValue);
inline void setup_results(Results *r);
inline bool city_is_long(PackedCity city);
void print256(__m256i var);

#if DEBUG
#define D(x) x
#define TIMER_RESET()  clock_gettime(CLOCK_MONOTONIC, &tic);
#define TIMER_MS(name) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-12s: %9.3f ms\n", name, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000);
#define TIMER_MS_NUM(name, n) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-9s %2d: %9.3f ms\n", name, n, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000);
//#define TIMER_US(name) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-12s: %9.3f us\n", name, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000000);
#define TIMER_US(name)
#define TIMER_INIT()   struct timespec tic, toc; (void)tic; (void)toc; TIMER_RESET();
#else
#define D(x)
#define TIMER_RESET()
#define TIMER_INIT()
#define TIMER_MS_NUM(name, n)
#define TIMER_MS(name)
#define TIMER_MS_NUM(name, n)
#define TIMER_US(name)
#endif

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define MIN(x, y) (y + ((x - y) & ((x - y) >> 31)))
#define MAX(x, y) (x - ((x - y) & ((x - y) >> 31)))

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

#define HASH_ENTRY_SIZE ((int)(STRIDE * sizeof(HashEntry)))

#define HASH_DATA_OFFSET 5        // log2(HASH_DATA_ENTRY_WIDTH)
#define HASH_CITY_OFFSET 5        // log2(SHORT_CITY_LENGTH)
#define HASH_CITY_LONG_OFFSET 7   // log2(LONG_CITY_LENGTH)

#define HASH_SHORT_MASK  (((1 << HASH_SHIFT       ) - 1) << MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))
#define HASH_LONG_MASK   (((1 << HASH_LONG_SHIFT  ) - 1) << HASH_CITY_LONG_OFFSET)
#define HASH_RESULT_MASK (((1 << HASH_RESULT_SHIFT) - 1) << HASH_CITY_OFFSET)

#define HASH_DATA_SHIFT (HASH_DATA_OFFSET - MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))
#define HASH_CITY_SHIFT (HASH_CITY_OFFSET - MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))

#define PACKED_OFFSETS_SIZE     PAGE_CEIL((int)sizeof(int) * MAX_CITIES)
#define HASHED_CITIES_SIZE      HUGE_PAGE_CEIL(SHORT_CITY_LENGTH * HASH_ENTRIES)
#define HASHED_DATA_SIZE        HUGE_PAGE_CEIL(HASH_ENTRY_SIZE   * HASH_ENTRIES)
#define HASHED_CITIES_LONG_SIZE HUGE_PAGE_CEIL(LONG_CITY_LENGTH  * HASH_LONG_ENTRIES)

#define HASH_MEMORY_SIZE       (PACKED_OFFSETS_SIZE + HASHED_CITIES_SIZE  + HASHED_DATA_SIZE + HASHED_CITIES_LONG_SIZE)

#define RESULTS_SIZE               LINE_CEIL(sizeof(Results))
#define RESULTS_REFS_SIZE          LINE_CEIL(sizeof(ResultsRef)  * MAX_CITIES)
#define RESULTS_ROWS_SIZE          LINE_CEIL(sizeof(ResultsRow)  * HASH_ENTRIES)
#define RESULTS_LONG_CITIES_SIZE   LINE_CEIL(sizeof(LongCity)    * MAX_CITIES)

#define RESULTS_MEMORY_SIZE        PAGE_CEIL(RESULTS_SIZE + RESULTS_REFS_SIZE + RESULTS_ROWS_SIZE + RESULTS_LONG_CITIES_SIZE)

#define MMAP_DATA_SIZE (1L << 32)
#define DUMMY_SIZE     PAGE_SIZE
#define TRAILING_SPACE PAGE_SIZE
#define MAX_CHUNK_SIZE (MMAP_DATA_SIZE - DUMMY_SIZE - TRAILING_SPACE)

#define SUM_BITS 35 // 1 + ceil(log2(1B * 999 / 8 / 8)
#define SUM_SIGN_BIT (1L << (SUM_BITS))
#define COUNT_BITS_START (SUM_BITS + 1)

#define EXTRACT_COUNT(v) ((int)(v >> COUNT_BITS_START))
#define SUM_MASK ((1L << COUNT_BITS_START) - 1)
#define EXTRACT_SUM(v) ((v & SUM_MASK) - SUM_SIGN_BIT)

#define LONG_CITY_SENTINEL 0xFACADE00

alignas(32) const void * const MASKED_DUMMY = (long []){
  'A' << 8 | 'D' << 16, 0, 0, 0
};

alignas(64) const void * const CITY_MASK = (long []){
  -1, -1, -1, -1, 0, 0, 0, 0,
};

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

  if ((fileStat.st_size - 1) / PAGE_SIZE < numWorkers) {
    D(fprintf(stderr, "decreasing numWorkers to %ld\n", fileStat.st_size / PAGE_SIZE + 1);)
    numWorkers = (int) (fileStat.st_size / PAGE_SIZE) + 1;
  }

  void *mem = mmap(NULL, RESULTS_MEMORY_SIZE + sizeof(Worker) * numWorkers, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Results *results = mem;
  setup_results(results);

  mem += RESULTS_MEMORY_SIZE;

  Worker *workers = mem;
  prep_workers(workers, numWorkers, warmup, fd, &fileStat);

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
      w->end = fileStat->st_size;
    }
    w->warmup = warmup;
  }
}


__attribute__((always_inline)) inline bool long_city_equal(LongCity *a, LongCity *b) {
  __m256i xor0 = _mm256_xor_si256(a->regs[0], b->regs[0]);
  __m256i xor1 = _mm256_xor_si256(a->regs[1], b->regs[1]);
  __m256i xor2 = _mm256_xor_si256(a->regs[2], b->regs[2]);
  __m256i xor3 = _mm256_xor_si256(a->regs[3], b->regs[3]);
  return _mm256_testz_si256(xor0, xor0) && _mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2) && _mm256_testz_si256(xor3, xor3);
}

void merge(Results * restrict dst, Results * restrict src) {
  for (int i = 0; i < src->numCities; i++) {
    ResultsRef ref = src->refs[i];
    ResultsRow row = src->rows[ref.offset / SHORT_CITY_LENGTH];
    int hashValue = hash_city(row.city.reg);

    if (unlikely(city_is_long(row.city))) {
      LongCity *longCity = src->longCities + row.city.longRef.index;

      int dstLongCityIdx = 0;
      LongCity *dstLongCity;
      for (; dstLongCityIdx < dst->numLongCities; dstLongCityIdx++) {
        dstLongCity = dst->longCities + dstLongCityIdx;
        if (long_city_equal(longCity, dstLongCity)) {
          break;
        }
      }
      // long city not in dst, insert it
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
        dstRow->sum += row.sum;
        dstRow->count += row.count;
        dstRow->min = MIN(dstRow->min, row.min);
        dstRow->max = MAX(dstRow->max, row.max);
        break;
      }

      if (_mm256_testz_si256(dstRow->city.reg, dstRow->city.reg)) {
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

void process(int id, Worker * workers, int numWorkers, int fdOut, Results *out) {
  TIMER_INIT();

  // processes at this point never process chunks, they only fork() child processes and then merge the results
  // assign children process to chunks if there are fewer than max_k workers to create
  // otherwise children process recurse to here and branch out again as necessary
  const int max_k = 8;
  const bool doWork = numWorkers <= max_k;
  const int k = doWork ? numWorkers : (numWorkers + (max_k - 1)) / max_k;

  int fd[k][2];
  struct pollfd poll_fds[k];

  void *tmp = mmap(NULL, RESULTS_MEMORY_SIZE * k, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  Results *childResults[k];
  for (int i = 0; i < k; i++) {
    childResults[i] = tmp + i* RESULTS_MEMORY_SIZE;
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

    int n = (numWorkers + ((k-i)/2)) / (k - i);
    numWorkers -= n;

    if (fork() == 0) {
      close(fd[i][0]);

      if (doWork) {
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

      process(new_id, workers, n, fd[i][1], childResults[i]);
      if (UNMAP) {
        while(wait(NULL) != -1) {}
      }
      exit(0);
    }
    new_id += n;
  }

  int childrenFinished = 0;
  while(childrenFinished < k) {
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


  if (fdOut != -1) {
    if (write(fdOut, "0", 1) < 0) {
      perror("parrent write");
      exit(1);
    }
    if (UNMAP) {
      while(wait(NULL) != -1) {}
    }
    exit(0);
  }

  if (UNMAP) {
    TIMER_RESET();
    while(wait(NULL) != -1) {}
    TIMER_MS("unmap");
  }
}

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

void convert_hash_to_results(Hash * restrict hash, Results * restrict out) {
  out->numCities = hash->counts.numCities;
  out->numLongCities = 0;

  for (int i = 0; i < hash->counts.numCities; i++) {
    int offset = hash->p.packedOffsets[i];
    PackedCity city = { .reg = _mm256_load_si256(hash->p.hashedCities + offset)};
    HashRow *rows = hash->p.hashedStorage + offset * (HASH_ENTRY_SIZE / SHORT_CITY_LENGTH);

    long sum  = EXTRACT_SUM(rows[0].packedSumCount);
    int count = EXTRACT_COUNT(rows[0].packedSumCount);
    int min   = rows[0].min;
    int max   = rows[0].max;

    for (int i = 1; i < STRIDE; i++) {
      sum +=  EXTRACT_SUM(rows[i].packedSumCount);
      count +=  EXTRACT_COUNT(rows[i].packedSumCount);
      min = MIN(min, rows[i].min);
      max = MAX(max, rows[i].max);
    }

    if (unlikely(city_is_long(city))) {
      LongCity *longCity = hash->p.hashedCitiesLong + city.longRef.index;
      out->longCities[out->numLongCities] = *longCity;

      city.longRef.index = out->numLongCities;
      out->numLongCities++;
    }

    offset = (offset >> (HASH_SHIFT - HASH_RESULT_SHIFT)) & HASH_RESULT_MASK;
    while (1) {
      if (_mm256_testz_si256(out->rows[offset / SHORT_CITY_LENGTH].city.reg, out->rows[offset / SHORT_CITY_LENGTH].city.reg)) {
        out->rows[offset / SHORT_CITY_LENGTH] = (ResultsRow) {city, sum, count, min, max};
        break;
      }
      offset += SHORT_CITY_LENGTH;
      offset &= HASH_RESULT_MASK;
    }
    out->refs[i] = (ResultsRef) {offset};
  }
}

void start_worker(Worker *w, Results *out) {
  TIMER_INIT();

  // 4k pages at the front for the offsets, 2MB pagesfor the everything else
  // pre-fault everything except the long cities
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

  void * const data = mmap(NULL, MMAP_DATA_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  // \0AD;0.0\n
  __m256i dummyData = _mm256_set1_epi64x(0x0A302E303B444100);
  for (int i = 0; i < DUMMY_SIZE; i += 32) {
    _mm256_store_si256(data + i, dummyData);
  }

  for (long start = w->start; start < w->end; start += MAX_CHUNK_SIZE) {
    long end = start + MAX_CHUNK_SIZE > w->end ? w->end : start + MAX_CHUNK_SIZE;

    bool first = w->first && start == w->start;
    bool last = w->last && end == w->end;

    uint32_t chunk_size = (uint32_t)(end - start);
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


    uint32_t offsets[STRIDE + 1];
    if (first) {
      offsets[0] = DUMMY_SIZE;
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

__attribute__((aligned(4096))) void process_chunk(const void * const restrict base, const uint32_t * offsets, Hash * restrict hashOut) {
  alignas(64) Hash hash = *hashOut;
  alignas(64) uint64_t nums[STRIDE];
  alignas(32) uint32_t starts[STRIDE];
  bool checkFinished;

  __m256i starts_v = _mm256_loadu_si256((__m256i *)offsets);
  __m256i ends_v = _mm256_loadu_si256((__m256i *)(offsets + 1));
  __m256i finished_v = _mm256_set1_epi32(0);

  __m256i atEndMask = _mm256_cmpeq_epi32(starts_v, ends_v);
  checkFinished = !_mm256_testz_si256(atEndMask, atEndMask);

  _mm256_store_si256((__m256i *)starts, starts_v);

  __m256i dummy = _mm256_load_si256(MASKED_DUMMY);
  _mm256_store_si256(hash.p.hashedCities + hash_city(dummy), dummy);

  while(1) {
    if (unlikely(checkFinished)) {
      finished_v = _mm256_or_si256(finished_v, atEndMask);

      if (unlikely(_mm256_movemask_epi8(finished_v) == 0xFFFFFFFF)) {
        hashOut->counts = hash.counts;
        return;
      }

      starts_v = _mm256_andnot_si256(finished_v, starts_v);
      ends_v = (__m256i)_mm256_blendv_ps((__m256)ends_v, (__m256)_mm256_set1_epi32(DUMMY_SIZE), (__m256)finished_v);

      // wtf, why is this like 10 slower than the masked store
      //_mm256_store_si256((__m256i *)starts, starts_v);
      _mm256_maskstore_epi32((int *)starts, finished_v, starts_v);
    }

    __m256i rawCity0 = _mm256_loadu_si256(base + starts[0]);
    __m256i rawCity1 = _mm256_loadu_si256(base + starts[1]);
    __m256i rawCity2 = _mm256_loadu_si256(base + starts[2]);
    __m256i rawCity3 = _mm256_loadu_si256(base + starts[3]);
    __m256i rawCity4 = _mm256_loadu_si256(base + starts[4]);
    __m256i rawCity5 = _mm256_loadu_si256(base + starts[5]);
    __m256i rawCity6 = _mm256_loadu_si256(base + starts[6]);
    __m256i rawCity7 = _mm256_loadu_si256(base + starts[7]);

    __m256i semicolons = _mm256_set1_epi8(';');
    int semicolonBytes0 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity0, semicolons)));
    int semicolonBytes1 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity1, semicolons)));
    int semicolonBytes2 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity2, semicolons)));
    int semicolonBytes3 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity3, semicolons)));
    int semicolonBytes4 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity4, semicolons)));
    int semicolonBytes5 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity5, semicolons)));
    int semicolonBytes6 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity6, semicolons)));
    int semicolonBytes7 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity7, semicolons)));

    // 127 keeps the the opcode small
    _mm_prefetch(base + starts[0] + semicolonBytes0 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[1] + semicolonBytes1 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[2] + semicolonBytes2 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[3] + semicolonBytes3 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[4] + semicolonBytes4 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[5] + semicolonBytes5 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[6] + semicolonBytes6 + 127, _MM_HINT_NTA);
    _mm_prefetch(base + starts[7] + semicolonBytes7 + 127, _MM_HINT_NTA);

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

    __m256i semicolons_v = _mm256_set_epi32(semicolonBytes7, semicolonBytes6, semicolonBytes5, semicolonBytes4, semicolonBytes3, semicolonBytes2, semicolonBytes1, semicolonBytes0);
    __m256i longCities = _mm256_cmpeq_epi32(semicolons_v, _mm256_set1_epi32(32));

    if (unlikely(!_mm256_testz_si256(longCities, longCities))) {
      if (semicolonBytes0 == 32) {
        maskedCity0 = process_long(base +  starts[0], &hash, &semicolonBytes0);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes0, 0);
      }
      if (semicolonBytes1 == 32) {
        maskedCity1 = process_long(base +  starts[1], &hash, &semicolonBytes1);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes1, 1);
      }
      if (semicolonBytes2 == 32) {
        maskedCity2 = process_long(base +  starts[2], &hash, &semicolonBytes2);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes2, 2);
      }
      if (semicolonBytes3 == 32) {
        maskedCity3 = process_long(base +  starts[3], &hash, &semicolonBytes3);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes3, 3);
      }
      if (semicolonBytes4 == 32) {
        maskedCity4 = process_long(base +  starts[4], &hash, &semicolonBytes4);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes4, 4);
      }
      if (semicolonBytes5 == 32) {
        maskedCity5 = process_long(base +  starts[5], &hash, &semicolonBytes5);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes5, 5);
      }
      if (semicolonBytes6 == 32) {
        maskedCity6 = process_long(base +  starts[6], &hash, &semicolonBytes6);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes6, 6);
      }
      if (semicolonBytes7 == 32) {
        maskedCity7 = process_long(base +  starts[7], &hash, &semicolonBytes7);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes7, 7);
      }
    }

    __m256i city_hashes = hash_cities(maskedCity0, maskedCity1, maskedCity2, maskedCity3, maskedCity4, maskedCity5, maskedCity6, maskedCity7);

    starts_v = _mm256_add_epi32(starts_v, semicolons_v);

    // shuffled order
    nums[0] = *(long *)(base + starts[0] + semicolonBytes0 - 2);
    nums[1] = *(long *)(base + starts[1] + semicolonBytes1 - 2);
    nums[2] = *(long *)(base + starts[4] + semicolonBytes4 - 2);
    nums[3] = *(long *)(base + starts[5] + semicolonBytes5 - 2);
    nums[4] = *(long *)(base + starts[2] + semicolonBytes2 - 2);
    nums[5] = *(long *)(base + starts[3] + semicolonBytes3 - 2);
    nums[6] = *(long *)(base + starts[6] + semicolonBytes6 - 2);
    nums[7] = *(long *)(base + starts[7] + semicolonBytes7 - 2);

    // nums: 0, 1, 4, 5
    __m256i nums_low = _mm256_load_si256((__m256i *)nums);

    // nums: 2, 3, 6, 7
    __m256i nums_high = _mm256_load_si256((__m256i *)(nums + 4));

    // bytes 0-3 and 4-7
    __m256i low_words = (__m256i) _mm256_shuffle_ps((__m256)nums_low, (__m256)nums_high, 0x88);
    __m256i high_words = (__m256i) _mm256_shuffle_ps((__m256)nums_low, (__m256)nums_high, 0xDD);

    // byte 2 is FF (always matches semicolon) to stop sign() from zeroring in the positive case
    // byte 3 is 00/FF for the minus sign, ready for masking
    __m256i minus_mask = _mm256_cmpeq_epi8(low_words, _mm256_set1_epi16(';' + ('-' << 8)));

    // bytes 3-6, for the positive cases with a digit in byte 3
    __m256i nums_low_left1 = _mm256_slli_epi64(nums_low, 8);
    __m256i nums_high_left1 = _mm256_slli_epi64(nums_high, 8);
    __m256i high_words_left1 = (__m256i) _mm256_shuffle_ps((__m256)nums_low_left1, (__m256)nums_high_left1, 0xDD);

    // no negative sign, left aligned so decimal is alway in byte 1 or 2
    __m256i nums_blended = (__m256i)_mm256_blendv_ps((__m256)high_words_left1, (__m256)high_words, (__m256)minus_mask);
    // 2 cycle stall

    // 6 bytes default added to line length
    starts_v = _mm256_add_epi32(starts_v, _mm256_set1_epi32(6));

    // extra 1 byte for minus sign
    __m256i minus_mask_shift = _mm256_srli_epi32(minus_mask, 31);

    // byte 3 FF matches newline X.X and -X.X cases
    __m256i newline_mask = _mm256_cmpeq_epi8(nums_blended, _mm256_set1_epi8('\n'));

    // 1 shorter line length for for X.X and -X.X cases (subtract it later)
    __m256i newline_mask_shift = _mm256_srli_epi32(newline_mask, 31);

    // shift words in X.X and -X.X cases to always have decimal in byte 2
    __m256i newline_shift = _mm256_slli_epi32(newline_mask_shift, 3);
    nums_blended = _mm256_sllv_epi32(nums_blended, newline_shift);

    // convert ascii to numbers, hide '.' with saturation
    __m256i numbers = _mm256_subs_epu8(nums_blended, _mm256_set1_epi8('0'));

    __m256i mulled = _mm256_madd_epi16(numbers, _mm256_set1_epi32(0x0001640a));
    // 5 cycle stall

    // store start of next line
    starts_v = _mm256_add_epi32(starts_v, minus_mask_shift);
    starts_v = _mm256_sub_epi32(starts_v, newline_mask_shift);
    _mm256_store_si256((__m256i *)(starts), starts_v);

    atEndMask = _mm256_cmpeq_epi32(starts_v, ends_v);
    checkFinished = !_mm256_testz_si256(atEndMask, atEndMask);

    mulled = _mm256_slli_epi32(mulled, 14);
    mulled = _mm256_srli_epi32(mulled, 22);
    __m256i final = _mm256_sign_epi32(mulled, minus_mask);

    // scale+offset the hashs in the store/load to avoid register intermediates
    // long instead of int to advoid unecessary sign extends
    long hash0 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 0), maskedCity0);
    __m128i vals0 = _mm_load_si128(hash.p.hashedStorage + 4*hash0 + 16*0);

    long hash4 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 2), maskedCity4);
    __m128i vals4 = _mm_load_si128(hash.p.hashedStorage + 4*hash4 + 16*4);

    long hash1 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 4), maskedCity1);
    __m128i vals1 = _mm_load_si128(hash.p.hashedStorage + 4*hash1 + 16*1);

    long hash5 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 6), maskedCity5);
    __m128i vals5 = _mm_load_si128(hash.p.hashedStorage + 4*hash5 + 16*5);

    long hash2 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 1), maskedCity2);
    __m128i vals2 = _mm_load_si128(hash.p.hashedStorage + 4*hash2 + 16*2);

    long hash6 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 3), maskedCity6);
    __m128i vals6 = _mm_load_si128(hash.p.hashedStorage + 4*hash6 + 16*6);

    long hash3 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 5), maskedCity3);
    __m128i vals3 = _mm_load_si128(hash.p.hashedStorage + 4*hash3 + 16*3);

    long hash7 = insert_city(&hash, _mm256_extract_epi32(city_hashes, 7), maskedCity7);
    __m128i vals7 = _mm_load_si128(hash.p.hashedStorage + 4*hash7 + 16*7);

    __m256i ae = _mm256_set_m128i(vals4, vals0);
    __m256i bf = _mm256_set_m128i(vals5, vals1);
    __m256i cg = _mm256_set_m128i(vals6, vals2);
    __m256i dh = _mm256_set_m128i(vals7, vals3);

    __m256i abef_low = _mm256_unpacklo_epi64(ae, bf);
    __m256i cdgh_low = _mm256_unpacklo_epi64(cg, dh);

    // A3 B3 A4 B4 | E3 F3 E4 F4
    __m256i abef_high = _mm256_unpackhi_epi32(ae, bf);
    __m256i cdgh_high = _mm256_unpackhi_epi32(cg, dh);

    __m256i mins = _mm256_unpacklo_epi64(abef_high, cdgh_high);
    __m256i maxs = _mm256_unpackhi_epi64(abef_high, cdgh_high);

    // shift and zero extend
    __m256i abef_shift = _mm256_set_epi64x(0x0707070707060504, 0x0303030303020100, 0x0707070707060504, 0x0303030303020100);
    __m256i final_abef = _mm256_shuffle_epi8(final, abef_shift);
    __m256i cdgh_shift = _mm256_set_epi64x(0x0F0F0F0F0F0E0D0C, 0x0B0B0B0B0B0A0908, 0x0F0F0F0F0F0E0D0C, 0x0B0B0B0B0B0A0908);
    __m256i final_cdgh = _mm256_shuffle_epi8(final, cdgh_shift);

    __m256i inc = _mm256_set1_epi64x(1L << COUNT_BITS_START);

    __m256i new_abef_low = _mm256_add_epi64(abef_low, final_abef);
    new_abef_low = _mm256_add_epi64(new_abef_low, inc);

    __m256i new_cdgh_low = _mm256_add_epi64(cdgh_low, final_cdgh);
    new_cdgh_low = _mm256_add_epi64(new_cdgh_low, inc);

    __m256i new_mins = _mm256_min_epi32(mins, final);
    __m256i new_maxs = _mm256_max_epi32(maxs, final);


    // A3 A4 B3 B4 | E3 E4 F3 F4
    __m256i new_abef_high = _mm256_unpacklo_epi32(new_mins, new_maxs);
    __m256i new_cdgh_high = _mm256_unpackhi_epi32(new_mins, new_maxs);

    __m256i new_ae = _mm256_unpacklo_epi64(new_abef_low, new_abef_high);
    __m256i new_bf = _mm256_unpackhi_epi64(new_abef_low, new_abef_high);
    __m256i new_cg = _mm256_unpacklo_epi64(new_cdgh_low, new_cdgh_high);
    __m256i new_dh = _mm256_unpackhi_epi64(new_cdgh_low, new_cdgh_high);

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

int hash_long(long x, long y) {
  long seed = 0x9e3779b97f4a7c15; // ~fxhash
  return ((_lrotl(x * seed, 5) ^ y) * seed) & HASH_LONG_MASK;
}

__m256i process_long(const void * const restrict start, Hash * restrict h, int * restrict semicolonBytesOut) {
  __m256i seg0 = _mm256_loadu_si256(start +  0);
  __m256i seg1 = _mm256_loadu_si256(start + 32);
  __m256i seg2 = _mm256_loadu_si256(start + 64);
  __m256i seg3 = _mm256_loadu_si256(start + 96);

  __m256i semicolons = _mm256_set1_epi8(';');
  int semicolonBytes1 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg1, semicolons)));
  int semicolonBytes2 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg2, semicolons)));
  int semicolonBytes3 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg3, semicolons)));

  int hash = hash_long(*(long *)start, *((long *)start + 1));

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

__attribute__((always_inline)) inline __m256i hash_cities(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {
  __m256i ab = _mm256_inserti128_si256(a, _mm256_castsi256_si128(b), 1);
  __m256i cd = _mm256_inserti128_si256(c, _mm256_castsi256_si128(d), 1);
  __m256i ef = _mm256_inserti128_si256(e, _mm256_castsi256_si128(f), 1);
  __m256i gh = _mm256_inserti128_si256(g, _mm256_castsi256_si128(h), 1);

  cd = _mm256_slli_si256(cd, 8);
  gh = _mm256_slli_si256(gh, 8);

  __m256i acbd = _mm256_blend_epi32(ab, cd, 0xCC);
  __m256i egfh = _mm256_blend_epi32(ef, gh, 0xCC);

  // preserve nibbles
  __m256i acbd2 = _mm256_srli_epi64(acbd, 28);
  __m256i egfh2 = _mm256_srli_epi64(egfh, 28);

  // A_C_D_B_
  // E_G_F_H_
  acbd = _mm256_xor_si256(acbd, acbd2);
  egfh = _mm256_xor_si256(egfh, egfh2);

  __m256i acegbdfh = (__m256i) _mm256_shuffle_ps((__m256)acbd, (__m256)egfh, 0x88);

  __m256i hash = _mm256_madd_epi16(acegbdfh, acegbdfh);
  __m256i hash_mask = _mm256_set1_epi32(HASH_SHORT_MASK);
  return _mm256_and_si256(hash, hash_mask);
}

__attribute__((always_inline)) inline int hash_city(__m256i str) {
  __m256i zero = _mm256_set1_epi32(0);
  __m256i hash = hash_cities(str, zero, zero, zero, zero, zero, zero, zero);
  return _mm256_extract_epi32(hash, 0);
}

__attribute__((always_inline)) inline long insert_city(Hash * restrict h, long hash, const __m256i maskedCity) {

  while (1) {
    __m256i stored = _mm256_load_si256(h->p.hashedCities + hash);
    __m256i xor = _mm256_xor_si256(maskedCity, stored);
    if (likely(_mm256_testz_si256(xor, xor))) {
      return hash;

    }
    if (_mm256_testz_si256(stored, stored)) {
      _mm256_store_si256(h->p.hashedCities + hash, maskedCity);
      h->p.packedOffsets[h->counts.numCities] = hash;
      h->counts.numCities += 1;

      __m256i initData = _mm256_set_epi32(MIN_TEMP, MAX_TEMP, SUM_SIGN_BIT >> 32, 0,
                                          MIN_TEMP, MAX_TEMP, SUM_SIGN_BIT >> 32, 0);
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

int insert_city_long1(Hash * restrict hash, int hash_value, __m256i seg0, __m256i seg1) {
  while (1) {
    __m256i stored0 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value +  0);
    __m256i stored1 = _mm256_loadu_si256(hash->p.hashedCitiesLong + hash_value + 32);
    __m256i xor0 = _mm256_xor_si256(stored0, seg0);
    __m256i xor1 = _mm256_xor_si256(stored1, seg1);

    if (_mm256_testz_si256(xor0, xor0) &&_mm256_testz_si256(xor1, xor1)) {
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

    if (_mm256_testz_si256(xor0, xor0) &&_mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2)) {
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

    if (_mm256_testz_si256(xor0, xor0) &&_mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2) && _mm256_testz_si256(xor3, xor3)) {
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

uint32_t find_next_row(const void *data, uint32_t offset) {
  __m256i newlines = _mm256_set1_epi8('\n');
  __m256i chars = _mm256_loadu_si256(data + offset);
  uint32_t bytes = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, newlines)));
  if (likely(bytes < 32)) {
    return offset + bytes + 1;
  }
  while (*((char *)data + offset + bytes) != '\n') {
    bytes++;
  }
  return offset + bytes + 1;
}

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

__attribute__((always_inline)) inline __m256i city_from_long_hash(int hashValue) {
  return _mm256_set_epi32(0, 0, 0, 0, 0, 0, hashValue, LONG_CITY_SENTINEL);
}

__attribute__((always_inline)) inline bool city_is_long(PackedCity city) {
  return city.longRef.sentinel == LONG_CITY_SENTINEL;
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
