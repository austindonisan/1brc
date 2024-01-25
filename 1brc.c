#define _DEFAULT_SOURCE 1
#include <fcntl.h>
#include <features.h>
#include <immintrin.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <threads.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

typedef struct {
  int64_t packed_sum;
  int32_t min;
  int32_t max;
} hash_entry_t;

typedef struct {
  char *packed_cities;
  char *hashed_cities;
  char *hashed_storage;
  char *packed_cities_long;
  char *hashed_cities_long;
  int num_cities;
  int num_cities_long;
} hash_t;

typedef struct {
  hash_t hash;
  long start;
  long end;
  int fd;
  int worker_id;
  bool fork;
  bool first;
  bool last;
} worker_t;

typedef struct {
  int64_t sum;
  int32_t count;
  int16_t min;
  int16_t max;
} result_data_t;

typedef struct {
  result_data_t *data;
  char *city;
  int cityLength;
} result_t;

typedef struct {
  int num_results;
  result_t *results[];
} results_t;

void prep_workers(void *shared, int num_workers, bool fork, int fd, struct stat *fileStat);
void process_threads(void * shared, int num_workers);
void process_forks(void * shared, int num_workers);
int start_worker(void *arg);
void process_chunk(const char * const restrict base, const unsigned int * offsets, hash_t * restrict h);
__m256i process_long(const char * start, hash_t *h, int *semicolonBytesOut);
inline __m256i hash_cities(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h);
inline int hash_city(__m256i str);
inline int insert_city(int id, int hash, const __m256i maskedCity, hash_t * h);
int insert_city_long(int hash, __m256i seg0, __m256i seg1, __m256i seg2, __m256i seg3, hash_t *h);
int get_offset(int id, int hash, const __m256i maskedCity, hash_t * h);
void merge(hash_t *h);
void merge2(hash_t *a, hash_t *b);
results_t *sort_results(hash_t *hash, void *mem);
int sort_result(const void *a, const void *b);
worker_t *get_worker(void *shared, int worker_id);
unsigned int find_next_row(const void *data, unsigned int offset);
void print_results(results_t *results, void *mem);
void debug_results(hash_t *hash);
void print256(__m256i var);

#define DEBUG 1

#if DEBUG
#define D(x) x
#define TIMER_RESET()  clock_gettime(CLOCK_MONOTONIC, &tic);
#define TIMER_MS(name) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-8s: %9.3f ms\n", name, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000);
#define TIMER_US(name) clock_gettime(CLOCK_MONOTONIC, &toc); fprintf(stderr, "%-8s: %9.3f us\n", name, ((toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000000);
#define TIMER_INIT()   struct timespec tic, toc; TIMER_RESET();
#else
#define D(x)
#define TIMER_RESET()
#define TIMER_INIT()
#define TIMER_MS(name)
#define TIMER_US(name)
#endif

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define MIN(x, y) (y + ((x - y) & ((x - y) >> 31)))
#define MAX(x, y) (x - ((x - y) & ((x - y) >> 31)))

#define PAGE_SIZE 4096
#define PAGE_MASK (~(PAGE_SIZE - 1))
#define PAGE_TRUNC(v) ((v) & (PAGE_MASK))
#define PAGE_CEIL(v)  (PAGE_TRUNC(v + PAGE_SIZE - 1))
#define PAGE_TRUNC_P(p) ((void *)PAGE_TRUNC((uintptr_t)p))
#define PAGE_CEIL_P(p) ((void *)PAGE_CEIL((uintptr_t)p))

#define LINE_SIZE 64
#define LINE_MASK (~(LINE_SIZE - 1))
#define LINE_TRUNC(v) ((v) & (LINE_MASK))
#define LINE_CEIL(v)  (LINE_TRUNC(v + LINE_SIZE - 1))

#define MAX_CITIES 10001 // + 1 for dummy city
#define MAX_TEMP 999
#define MIN_TEMP -999

#define SHORT_CITY_LENGTH 32
#define LONG_CITY_LENGTH 128

#define STRIDE 8
#define HASH_ENTRY_SIZE (STRIDE * sizeof(hash_entry_t))

#define HASH_DATA_OFFSET 5        // log2(HASH_DATA_ENTRY_WIDTH)
#define HASH_CITY_OFFSET 5        // log2(SHORT_CITY_LENGTH)
#define HASH_CITY_LONG_OFFSET 7   // log2(LONG_CITY_LENGTH)

#define HASH_SHIFT 15              // max(desired, log2(MAX_CITIES))
#define HASH_LONG_SHIFT 14         // max(desired, log2(MAX_CITIES))

#define HASH_SHORT_MASK (((1 << HASH_SHIFT)      - 1) << MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))
#define HASH_LONG_MASK  (((1 << HASH_LONG_SHIFT) - 1) << HASH_CITY_LONG_OFFSET)

#define HASH_DATA_SHIFT (HASH_DATA_OFFSET - MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))
#define HASH_CITY_SHIFT (HASH_CITY_OFFSET - MIN(HASH_DATA_OFFSET, HASH_CITY_OFFSET))

#define HASH_LENGTH      (1 << HASH_SHIFT)
#define HASH_LONG_LENGTH (1 << HASH_LONG_SHIFT)

#define WORKER_SIZE             LINE_CEIL(sizeof(worker_t))
#define PACKED_CITIES_SIZE      LINE_CEIL(SHORT_CITY_LENGTH * MAX_CITIES)
#define HASHED_CITIES_SIZE      LINE_CEIL(SHORT_CITY_LENGTH * HASH_LENGTH)
#define HASHED_DATA_SIZE        LINE_CEIL(HASH_ENTRY_SIZE   * HASH_LENGTH)
#define PACKED_CITIES_LONG_SIZE LINE_CEIL(LONG_CITY_LENGTH  * MAX_CITIES)
#define HASHED_CITIES_LONG_SIZE LINE_CEIL(LONG_CITY_LENGTH  * HASH_LONG_LENGTH)

#define WORKER_MEMORY_SIZE PAGE_CEIL(WORKER_SIZE + PACKED_CITIES_SIZE + HASHED_CITIES_SIZE + HASHED_DATA_SIZE + PACKED_CITIES_LONG_SIZE + HASHED_CITIES_LONG_SIZE)

#define MMAP_DATA_SIZE (1L << 32)
#define MAX_CHUNK_SIZE (MMAP_DATA_SIZE - 2*PAGE_SIZE)

#define SUM_BITS 35 // 1 + ceil(log2(1B * 999 / 8 / 8)
#define SUM_SIGN_BIT (1L << (SUM_BITS))
#define COUNT_BITS_START (SUM_BITS + 1)

#define EXTRACT_COUNT(v) ((int)(v >> COUNT_BITS_START))
#define SUM_MASK ((1L << COUNT_BITS_START) - 1)
#define EXTRACT_SUM(v) ((v & SUM_MASK) - SUM_SIGN_BIT)

alignas(32) const char * const masked_dummy = (char []){
  0 ,'A','D', 0 , 0 , 0 , 0 , 0 ,
  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
  0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
 };

alignas(64) const char * const city_mask = (char []){
   1,  1,  1,  1,  1,  1,  1,  1,
   1,  1,  1,  1,  1,  1,  1,  1,
   1,  1,  1,  1,  1,  1,  1,  1,
   1,  1,  1,  1,  1,  1,  1,  1,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
 };

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "Usage: good file workers [fork]\n");
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

  int num_workers = atoi(argv[2]);
  if (num_workers < 1 || num_workers > 256) {
    fprintf(stderr, "workers must be between 1 and 256\n");
    return EXIT_FAILURE;
  }

  const bool use_fork = argc < 4 ? false : atoi(argv[3]) != 0;

  if ((fileStat.st_size - 1) / PAGE_SIZE < num_workers) {
    D(fprintf(stderr, "decreasing num_workers to %ld\n", fileStat.st_size / PAGE_SIZE + 1);)
    num_workers = (int) (fileStat.st_size / PAGE_SIZE) + 1;
  }

  void * shared_mem = mmap(NULL, WORKER_MEMORY_SIZE * (num_workers > 3 ? num_workers : 3), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  prep_workers(shared_mem, num_workers, use_fork, fd, &fileStat);

  if (num_workers == 1 && false) {
    start_worker((void *)get_worker(shared_mem, 0));
  }
  else if (use_fork) {
    process_forks(shared_mem, num_workers);
  }
  else {
    process_threads(shared_mem, num_workers);
  }

  hash_t *hash = &get_worker(shared_mem, 0)->hash;
  D(debug_results(hash));
  for (int i = 1; i < num_workers; i++) {
    merge2(hash, &get_worker(shared_mem, i)->hash);
  }

  TIMER_INIT();
  results_t *results = sort_results(hash, (void *)get_worker(shared_mem, 2));
  TIMER_US("sort");

  TIMER_RESET();
  print_results(results, (void *)get_worker(shared_mem, 1));
  TIMER_US("print");

  D(debug_results(hash));

  return 0;
}

void prep_workers(void *shared, int num_workers, bool fork, int fd, struct stat *fileStat) {
  long start = 0;
  long delta = PAGE_TRUNC(fileStat->st_size / num_workers);
  for (int i = 0; i < num_workers; i++) {
    worker_t *w = get_worker(shared, i);
    w->worker_id = i;
    w->fd = fd;
    w->start = start;
    w->end = (start += delta);
    w->first = i == 0;
    w->last = i == num_workers - 1;
    if (w->last) {
      w->end = fileStat->st_size;
    }
    w->fork = fork;
    w->hash.num_cities = 0;

    char *p = (char *)w + WORKER_SIZE;

    w->hash.packed_cities = p;
    p += PACKED_CITIES_SIZE;

    w->hash.hashed_cities = p;
    p += HASHED_CITIES_SIZE;

    w->hash.hashed_storage = p;
    p += HASHED_DATA_SIZE;

    w->hash.packed_cities_long = p;
    p += PACKED_CITIES_LONG_SIZE;

    w->hash.hashed_cities_long = p;
    p += HASHED_CITIES_LONG_SIZE;
  }
}

void process_threads(void * shared, int num_workers) {
  TIMER_INIT();

  thrd_t threads[num_workers];
  for (int i = 0; i < num_workers; i++) {
    worker_t *w = get_worker(shared, i);
    thrd_create(&threads[i], start_worker, w);
  }

  for (int i = 0; i < num_workers; i++) {
    thrd_join(threads[i], NULL);
  }

  TIMER_MS("done");
  return;
}

void process_forks(void *shared, int num_workers) {
  for (int i = 0; i < num_workers; i++) {
    if (fork() == 0) {
      start_worker((void *)get_worker(shared, i));
      exit(0);
    }
  }

  while(wait(NULL) != -1);
}

int start_worker(void *arg) {
  worker_t *w = (worker_t *)arg;

  TIMER_INIT();
  void * const data = mmap(NULL, MMAP_DATA_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  // \0AD;0.0\n
  __m256i dummyData = _mm256_set1_epi64x(0x0A302E303B444100);
  for (int i = 0; i < PAGE_SIZE; i += 32) {
    _mm256_store_si256(data + i, dummyData);
  }

  TIMER_MS("mmap 1");


  for (long start = w->start; start < w->end; start += MAX_CHUNK_SIZE) {

    long end = start + MAX_CHUNK_SIZE > w->end ? w->end : start + MAX_CHUNK_SIZE;

    bool first = w->first && start == w->start;
    bool last = w->last && end == w->end;

    unsigned int chunk_size = (unsigned int)(end - start);
    unsigned int mapped_file_length = last ? PAGE_CEIL(chunk_size) : chunk_size + PAGE_SIZE;

    TIMER_RESET();
    mmap(data + PAGE_SIZE, mapped_file_length, PROT_READ, MAP_SHARED | MAP_FIXED | (w->fork ? MAP_POPULATE : 0), w->fd, start);
    TIMER_MS("mmap 2");

    if (!w->fork) {
      TIMER_RESET();
      long dummy = 0;
      for (long i = PAGE_SIZE; i - PAGE_SIZE < mapped_file_length; i += PAGE_SIZE) {
        dummy += *(long *)(data + i);
      }
      __asm__ volatile( "" : : [dummy] "r" (dummy));
      TIMER_MS("warmup");
    }

    unsigned int offsets[STRIDE + 1];
    if (first) {
      offsets[0] = PAGE_SIZE;
    }
    for (int i = first ? 1 : 0; i < STRIDE; i++) {
      offsets[i] = find_next_row(data, chunk_size / STRIDE * i + PAGE_SIZE);
    }
    offsets[STRIDE] = last ? chunk_size + PAGE_SIZE : find_next_row(data, chunk_size + PAGE_SIZE);

    TIMER_RESET();
    process_chunk(data, offsets, &w->hash);
    TIMER_MS("chunk");
  }


  TIMER_RESET();
  merge(&w->hash);
  TIMER_US("merge");

  TIMER_RESET();
  D(munmap(data, MMAP_DATA_SIZE));
  TIMER_MS("munmap");

  return 0;
}

void process_chunk(const char * const restrict base, const unsigned int * offsets, hash_t * restrict h) {
  char * const values_map = h->hashed_storage;

  alignas(64) long nums[STRIDE];
  alignas(32) unsigned int starts[STRIDE];
  alignas(32) unsigned int ends[STRIDE];
  alignas(32) int finished[STRIDE] = {0};

  for (int i = 0; i < STRIDE; i++) {
    starts[i] = offsets[i];
    ends[i] = offsets[i + 1];
  }

  __m256i starts_v = _mm256_load_si256((__m256i *)starts);

  insert_city(0, hash_city(_mm256_loadu_si256((__m256i *)masked_dummy)), _mm256_loadu_si256((__m256i *)masked_dummy), h);

  while(1) {
    __m256i ends_v =  _mm256_load_si256((__m256i *)ends);
    __m256i at_end_mask = _mm256_cmpeq_epi32(starts_v, ends_v);
    if (unlikely(_mm256_movemask_epi8(at_end_mask))) {

      __m256i finished_v = _mm256_load_si256((__m256i *)finished);
      finished_v = _mm256_or_si256(finished_v, at_end_mask);

      if (unlikely(_mm256_movemask_epi8(finished_v) == 0xFFFFFFFF)) {
        return;
      }

      starts_v = _mm256_srlv_epi32(starts_v, finished_v);
      _mm256_store_si256((__m256i *)finished, finished_v);

      // wtf, why is this like 10 slower than the masked store
      //_mm256_store_si256((__m256i *)starts, starts_v);

      _mm256_maskstore_epi32((int *)starts, finished_v, _mm256_set1_epi32(0));
      _mm256_maskstore_epi32((int *)ends, finished_v, _mm256_set1_epi32(PAGE_SIZE));
    }

    __m256i rawCity0 = _mm256_loadu_si256((__m256i *)(base + starts[0]));
    __m256i rawCity1 = _mm256_loadu_si256((__m256i *)(base + starts[1]));
    __m256i rawCity2 = _mm256_loadu_si256((__m256i *)(base + starts[2]));
    __m256i rawCity3 = _mm256_loadu_si256((__m256i *)(base + starts[3]));
    __m256i rawCity4 = _mm256_loadu_si256((__m256i *)(base + starts[4]));
    __m256i rawCity5 = _mm256_loadu_si256((__m256i *)(base + starts[5]));
    __m256i rawCity6 = _mm256_loadu_si256((__m256i *)(base + starts[6]));
    __m256i rawCity7 = _mm256_loadu_si256((__m256i *)(base + starts[7]));

    int semicolonBytes0 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity0, _mm256_set1_epi8(';'))));
    int semicolonBytes1 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity1, _mm256_set1_epi8(';'))));
    int semicolonBytes2 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity2, _mm256_set1_epi8(';'))));
    int semicolonBytes3 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity3, _mm256_set1_epi8(';'))));
    int semicolonBytes4 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity4, _mm256_set1_epi8(';'))));
    int semicolonBytes5 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity5, _mm256_set1_epi8(';'))));
    int semicolonBytes6 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity6, _mm256_set1_epi8(';'))));
    int semicolonBytes7 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(rawCity7, _mm256_set1_epi8(';'))));

    __m256i rawMask0 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes0));
    __m256i rawMask1 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes1));
    __m256i rawMask2 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes2));
    __m256i rawMask3 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes3));
    __m256i rawMask4 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes4));
    __m256i rawMask5 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes5));
    __m256i rawMask6 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes6));
    __m256i rawMask7 = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes7));

    __m256i maskedCity0 = _mm256_sign_epi8(rawCity0, rawMask0);
    __m256i maskedCity1 = _mm256_sign_epi8(rawCity1, rawMask1);
    __m256i maskedCity2 = _mm256_sign_epi8(rawCity2, rawMask2);
    __m256i maskedCity3 = _mm256_sign_epi8(rawCity3, rawMask3);
    __m256i maskedCity4 = _mm256_sign_epi8(rawCity4, rawMask4);
    __m256i maskedCity5 = _mm256_sign_epi8(rawCity5, rawMask5);
    __m256i maskedCity6 = _mm256_sign_epi8(rawCity6, rawMask6);
    __m256i maskedCity7 = _mm256_sign_epi8(rawCity7, rawMask7);

    __m256i semicolons_v = _mm256_set_epi32(semicolonBytes7, semicolonBytes6, semicolonBytes5, semicolonBytes4, semicolonBytes3, semicolonBytes2, semicolonBytes1, semicolonBytes0);
    if (unlikely(_mm256_movemask_epi8(_mm256_cmpeq_epi32(semicolons_v, _mm256_set1_epi32(32))))) {
      if (semicolonBytes0 == 32) {
        maskedCity0 = process_long(base +  starts[0], h, &semicolonBytes0);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes0, 0);
      }
      if (unlikely(semicolonBytes1 == 32)) {
        maskedCity1 = process_long(base +  starts[1], h, &semicolonBytes1);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes1, 1);
      }
      if (unlikely(semicolonBytes2 == 32)) {
        maskedCity2 = process_long(base +  starts[2], h, &semicolonBytes2);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes2, 2);
      }
      if (unlikely(semicolonBytes3 == 32)) {
        maskedCity3 = process_long(base +  starts[3], h, &semicolonBytes3);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes3, 3);
      }
      if (unlikely(semicolonBytes4 == 32)) {
        maskedCity4 = process_long(base +  starts[4], h, &semicolonBytes4);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes4, 4);
      }
      if (unlikely(semicolonBytes5 == 32)) {
        maskedCity5 = process_long(base +  starts[5], h, &semicolonBytes5);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes5, 5);
      }
      if (unlikely(semicolonBytes6 == 32)) {
        maskedCity6 = process_long(base +  starts[6], h, &semicolonBytes6);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes6, 6);
      }
      if (unlikely(semicolonBytes7 == 32)) {
        maskedCity7 = process_long(base +  starts[7], h, &semicolonBytes7);
        semicolons_v = _mm256_insert_epi32(semicolons_v, semicolonBytes7, 7);
      }
    }

    __m256i city_hashes = hash_cities(maskedCity0, maskedCity1, maskedCity2, maskedCity3, maskedCity4, maskedCity5, maskedCity6, maskedCity7);

    starts_v = _mm256_add_epi32(starts_v, semicolons_v);

    // shuffled order
    nums[0] = *(long *)(base + starts[0] + semicolonBytes0);
    nums[1] = *(long *)(base + starts[1] + semicolonBytes1);
    nums[2] = *(long *)(base + starts[4] + semicolonBytes4);
    nums[3] = *(long *)(base + starts[5] + semicolonBytes5);
    nums[4] = *(long *)(base + starts[2] + semicolonBytes2);
    nums[5] = *(long *)(base + starts[3] + semicolonBytes3);
    nums[6] = *(long *)(base + starts[6] + semicolonBytes6);
    nums[7] = *(long *)(base + starts[7] + semicolonBytes7);

    // 0, 1, 4, 5
    __m256i nums_low = _mm256_load_si256((__m256i *)nums);

    // 2, 3, 6, 7
    __m256i nums_high = _mm256_load_si256((__m256i *)(nums + 4));

    __m256i low_period_mask =  _mm256_cmpeq_epi8(nums_low, _mm256_set1_epi8('.'));
    __m256i high_period_mask =  _mm256_cmpeq_epi8(nums_high, _mm256_set1_epi8('.'));

    __m256i low_words = (__m256i) _mm256_shuffle_ps((__m256)nums_low, (__m256)nums_high, 0x88);

    __m256i semicolon_mask = _mm256_set1_epi64x(';');
    nums_low =  _mm256_xor_si256(nums_low, semicolon_mask);
    nums_high =  _mm256_xor_si256(nums_high, semicolon_mask);

    // remove high 7/8 bits for -99.9 case
    low_period_mask =  _mm256_slli_epi64(low_period_mask, 31);
    high_period_mask =  _mm256_slli_epi64(high_period_mask, 31);

    // remove low 7/8 bits for 99.9 case
    low_period_mask =  _mm256_srli_epi64(low_period_mask, 62);
    high_period_mask =  _mm256_srli_epi64(high_period_mask, 62);

    // 0x88 0 1 4 5, 2 3 6 7 -> 0L 1L 2L 3L 4L 5L 6L 7L
    __m256i period_offsets = (__m256i) _mm256_shuffle_ps((__m256)low_period_mask, (__m256)high_period_mask, 0x88);

    period_offsets = _mm256_add_epi32(period_offsets, _mm256_set1_epi32(5));
    starts_v = _mm256_add_epi32(starts_v, period_offsets);
    _mm256_store_si256((__m256i *)(starts), starts_v);

    __m256i load3 = _mm256_set_epi64x( 0x800F0D0C800B0908, 0x8007050480030100, 0x800F0D0C800B0908, 0x8007050480030100);
    low_period_mask =  _mm256_slli_epi64(low_period_mask, 3);
    high_period_mask =  _mm256_slli_epi64(high_period_mask, 3);
    __m256i low_shifted =_mm256_srlv_epi64(nums_low, low_period_mask);
    __m256i high_shifted =_mm256_srlv_epi64(nums_high, high_period_mask);
    __m256i numbers = (__m256i) _mm256_shuffle_ps((__m256)low_shifted, (__m256)high_shifted, 0x88);
    numbers = _mm256_shuffle_epi8(numbers, load3);

    // convert from ascii, hide '-'
    numbers = _mm256_subs_epu8(numbers, _mm256_set1_epi8('0'));

    __m256i mulled = _mm256_madd_epi16(numbers, _mm256_set1_epi32(0x0100640a));
    mulled = _mm256_slli_epi32(mulled, 14);
    mulled = _mm256_srli_epi32(mulled, 22);

    __m256i minus = _mm256_set1_epi16('-' << 8 | ';');
    __m256i minus_mask =  _mm256_cmpeq_epi8(low_words, minus);
    __m256i shifted_minus_mask = _mm256_slli_epi32(minus_mask, 16);
    __m256i final = _mm256_sign_epi32(mulled, shifted_minus_mask);

    int offset0 = insert_city(0, _mm256_extract_epi32(city_hashes, 0), maskedCity0, h);
    int offset1 = insert_city(1, _mm256_extract_epi32(city_hashes, 4), maskedCity1, h);
    int offset2 = insert_city(2, _mm256_extract_epi32(city_hashes, 1), maskedCity2, h);
    int offset3 = insert_city(3, _mm256_extract_epi32(city_hashes, 5), maskedCity3, h);
    int offset4 = insert_city(4, _mm256_extract_epi32(city_hashes, 2), maskedCity4, h);
    int offset5 = insert_city(5, _mm256_extract_epi32(city_hashes, 6), maskedCity5, h);
    int offset6 = insert_city(6, _mm256_extract_epi32(city_hashes, 3), maskedCity6, h);
    int offset7 = insert_city(7, _mm256_extract_epi32(city_hashes, 7), maskedCity7, h);

    __m256i ae = _mm256_set_m128i(_mm_load_si128((__m128i *)(values_map + offset4)), _mm_load_si128((__m128i *)(values_map + offset0)));
    __m256i bf = _mm256_set_m128i(_mm_load_si128((__m128i *)(values_map + offset5)), _mm_load_si128((__m128i *)(values_map + offset1)));
    __m256i cg = _mm256_set_m128i(_mm_load_si128((__m128i *)(values_map + offset6)), _mm_load_si128((__m128i *)(values_map + offset2)));
    __m256i dh = _mm256_set_m128i(_mm_load_si128((__m128i *)(values_map + offset7)), _mm_load_si128((__m128i *)(values_map + offset3)));


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

    _mm_store_si128((__m128i *)(values_map + offset0), _mm256_extracti128_si256(new_ae, 0));
    _mm_store_si128((__m128i *)(values_map + offset1), _mm256_extracti128_si256(new_bf, 0));
    _mm_store_si128((__m128i *)(values_map + offset2), _mm256_extracti128_si256(new_cg, 0));
    _mm_store_si128((__m128i *)(values_map + offset3), _mm256_extracti128_si256(new_dh, 0));
    _mm_store_si128((__m128i *)(values_map + offset4), _mm256_extracti128_si256(new_ae, 1));
    _mm_store_si128((__m128i *)(values_map + offset5), _mm256_extracti128_si256(new_bf, 1));
    _mm_store_si128((__m128i *)(values_map + offset6), _mm256_extracti128_si256(new_cg, 1));
    _mm_store_si128((__m128i *)(values_map + offset7), _mm256_extracti128_si256(new_dh, 1));
  }
}

unsigned long rotate_long(unsigned long x) {
  return (x << 5) | (x >> (x - 5));
}
int hash_long(long x, long y) {
  long seed = 0x9e3779b97f4a7c15;
  return ((rotate_long(x * seed) ^ y) * seed) & HASH_LONG_MASK;
}
__m256i process_long(const char * start, hash_t *h, int *semicolonBytesOut) {
  __m256i seg0 = _mm256_loadu_si256((__m256i *)start);
  __m256i seg1 = _mm256_loadu_si256((__m256i *)start + 1);
  __m256i seg2 = _mm256_loadu_si256((__m256i *)start + 2);
  __m256i seg3 = _mm256_loadu_si256((__m256i *)start + 3);
  int semicolonBytes1 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg1, _mm256_set1_epi8(';'))));
  int semicolonBytes2 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg2, _mm256_set1_epi8(';'))));
  int semicolonBytes3 = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg3, _mm256_set1_epi8(';'))));

  if (semicolonBytes1 < 32) {
    *semicolonBytesOut = 32 + semicolonBytes1;
    __m256i mask = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes1));
    seg1 = _mm256_sign_epi8(seg1, mask);
    seg2 = _mm256_set1_epi8(0);
    seg3 = _mm256_set1_epi8(0);
  }
  else if (semicolonBytes2 < 32) {
    *semicolonBytesOut = 64 + semicolonBytes2;
    __m256i mask = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes2));
    seg2 = _mm256_sign_epi8(seg2, mask);
    seg3 = _mm256_set1_epi8(0);
  }
  else {
    *semicolonBytesOut = 96 + semicolonBytes3;
    __m256i mask = _mm256_loadu_si256((__m256i *)(city_mask + 32 - semicolonBytes3));
    seg3 = _mm256_sign_epi8(seg3, mask);
  }

  int hash = hash_long(*(long *)start, *((long *)start + 1));
  int short_hash = insert_city_long(hash, seg0, seg1, seg2, seg3, h);
  return _mm256_slli_si256(_mm256_set1_epi32(short_hash), 4);
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
  __m256i acbd2 = _mm256_srli_epi64(acbd, 31);
  __m256i egfh2 = _mm256_srli_epi64(egfh, 31);

  // A_C_D_B_
  // E_G_F_H_
  acbd = _mm256_xor_si256(acbd, acbd2);
  egfh = _mm256_xor_si256(egfh, egfh2);

  __m256i acegbdfh = (__m256i) _mm256_shuffle_ps((__m256)acbd, (__m256)egfh, 0x88);

  __m256i hash = _mm256_madd_epi16(acegbdfh, acegbdfh);
  hash = _mm256_xor_si256(hash, acegbdfh);
  __m256i hash_mask = _mm256_set1_epi32(HASH_SHORT_MASK);
  return _mm256_and_si256(hash, hash_mask);
}

__attribute__((always_inline)) inline int hash_city(__m256i str) {
  __m256i zero = _mm256_set1_epi32(0);
  __m256i hash = hash_cities(str, zero, zero, zero, zero, zero, zero, zero);
  return _mm256_extract_epi32(hash, 0);
}

__attribute__((always_inline)) inline int insert_city(int id, int hash, const __m256i maskedCity, hash_t * h) {
  while (1) {
    __m256i stored = _mm256_load_si256((__m256i *)(h->hashed_cities + hash + 0));

    __m256i xor = _mm256_xor_si256(maskedCity, stored);
    if (likely(_mm256_testz_si256(xor, xor))) {
      return hash * 4 + id * 16;
    }
    if (_mm256_testz_si256(stored, stored)) {
      _mm256_store_si256((__m256i *)(h->packed_cities + h->num_cities * SHORT_CITY_LENGTH), maskedCity);
      _mm256_store_si256((__m256i *)(h->hashed_cities + hash), maskedCity);
      h->num_cities += 1;

      for (int i = 0; i < STRIDE; i++) {
        ((long*)(h->hashed_storage + hash * 4 + i * 16))[0] = SUM_SIGN_BIT;
        ((int*)(h->hashed_storage + hash * 4 + i * 16))[2] = MAX_TEMP;
        ((int*)(h->hashed_storage + hash * 4 + i * 16))[3] = MIN_TEMP;
      }
      return hash * 4 + id * 16;
    }
    hash += SHORT_CITY_LENGTH;
  }
}

__attribute__((always_inline)) inline int get_offset(int id, int hash, const __m256i maskedCity, hash_t * h) {
  while (1) {
    __m256i stored = _mm256_load_si256((__m256i *)(h->hashed_cities + hash));
    __m256i xor = _mm256_xor_si256(maskedCity, stored);
    if (likely(_mm256_testz_si256(xor, xor))) {
      return hash * 4 + id * 16;
    }
    if (_mm256_testz_si256(stored, stored)) {
      return -1;
    }
    hash += SHORT_CITY_LENGTH;
  }
}

int insert_city_long(int hash, __m256i seg0, __m256i seg1, __m256i seg2, __m256i seg3, hash_t *h) {
  while (1) {
    __m256i stored0 = _mm256_loadu_si256((__m256i *)(h->hashed_cities_long + hash));
    __m256i stored1 = _mm256_loadu_si256((__m256i *)(h->hashed_cities_long + hash) + 1);
    __m256i stored2 = _mm256_loadu_si256((__m256i *)(h->hashed_cities_long + hash) + 2);
    __m256i stored3 = _mm256_loadu_si256((__m256i *)(h->hashed_cities_long + hash) + 3);
    __m256i xor0 = _mm256_xor_si256(stored0, seg0);
    __m256i xor1 = _mm256_xor_si256(stored1, seg1);
    __m256i xor2 = _mm256_xor_si256(stored2, seg2);
    __m256i xor3 = _mm256_xor_si256(stored3, seg3);

      if (_mm256_testz_si256(xor0, xor0) &&_mm256_testz_si256(xor1, xor1) && _mm256_testz_si256(xor2, xor2) && _mm256_testz_si256(xor3, xor3)) {
      return hash;
    }
    if (_mm256_testz_si256(stored0, stored0)) {
      _mm256_store_si256((__m256i *)(h->packed_cities_long + h->num_cities_long * LONG_CITY_LENGTH), seg0);
      _mm256_store_si256((__m256i *)(h->packed_cities_long + h->num_cities_long * LONG_CITY_LENGTH) + 1, seg1);
      _mm256_store_si256((__m256i *)(h->packed_cities_long + h->num_cities_long * LONG_CITY_LENGTH) + 2, seg2);
      _mm256_store_si256((__m256i *)(h->packed_cities_long + h->num_cities_long * LONG_CITY_LENGTH) + 3, seg3);

      _mm256_store_si256((__m256i *)(h->hashed_cities_long + hash), seg0);
      _mm256_store_si256((__m256i *)(h->hashed_cities_long + hash) + 1, seg1);
      _mm256_store_si256((__m256i *)(h->hashed_cities_long + hash) + 2, seg2);
      _mm256_store_si256((__m256i *)(h->hashed_cities_long + hash) + 3, seg3);
      h->num_cities_long++;
      return hash;
    }
    hash += LONG_CITY_LENGTH;
  }
}

void merge(hash_t *h) {
  for (int i = 0; i < h->num_cities; i++) {
    __m256i city = _mm256_load_si256((__m256i *)(h->packed_cities + i * SHORT_CITY_LENGTH));
    int hash = hash_city(city);
    int off = get_offset(0, hash, city, h);

    unsigned long psc = *(long *)(h->hashed_storage + off);
    long sum  = EXTRACT_SUM(psc);
    int count = EXTRACT_COUNT(psc);
    int min   = *(int *)(h->hashed_storage + off + 8);
    int max   = *(int *)(h->hashed_storage + off + 12);

    for (int i = 1; i < STRIDE; i++) {
      unsigned long packed_sum_count = *(long *)(h->hashed_storage + off + i * 16 + 0);
      sum +=  EXTRACT_SUM(packed_sum_count);
      count +=  EXTRACT_COUNT(packed_sum_count);
      int old_min = *(int *)(h->hashed_storage + off + i * 16 + 8);
      int old_max = *(int *)(h->hashed_storage + off + i * 16 + 12);
      min = MIN(min, old_min);
      max = MAX(max, old_max);
    }
    *(long  *)(h->hashed_storage + off +  0)  = sum;
    *(int   *)(h->hashed_storage + off +  8)  = count;
    *(short *)(h->hashed_storage + off + 12)  = min;
    *(short *)(h->hashed_storage + off + 14)  = max;
  }
}

void merge2(hash_t *a, hash_t *b) {
  for (int i = 0; i < b->num_cities; i++) {
    __m256i city = _mm256_load_si256((__m256i *)(b->packed_cities + i * SHORT_CITY_LENGTH));
    int hash = hash_city(city);
    int b_offset  = get_offset(0, hash, city, b);
    int a_offset = get_offset(0, hash, city, a);
    if (likely(a_offset != -1)) {
      *(long  *)(a->hashed_storage + a_offset +  0) += *(long  *)(b->hashed_storage + b_offset +  0);
      *(int   *)(a->hashed_storage + a_offset +  8) += *(int   *)(b->hashed_storage + b_offset +  8);
      *(short *)(a->hashed_storage + a_offset + 12) = MIN(*(short *)(a->hashed_storage + a_offset + 12), *(short *)(b->hashed_storage + b_offset + 12));
      *(short *)(a->hashed_storage + a_offset + 14) = MAX(*(short *)(a->hashed_storage + a_offset + 14), *(short *)(b->hashed_storage + b_offset + 14));
    }
    else {
      a_offset = insert_city(0, hash, city, a);
      *(long  *)(a->hashed_storage + a_offset +  0) = *(long  *)(b->hashed_storage + b_offset +  0);
      *(int   *)(a->hashed_storage + a_offset +  8) = *(int   *)(b->hashed_storage + b_offset +  8);
      *(short *)(a->hashed_storage + a_offset + 12) = *(short *)(b->hashed_storage + b_offset + 12);
      *(short *)(a->hashed_storage + a_offset + 14) = *(short *)(b->hashed_storage + b_offset + 14);
    }
  }
}

results_t *sort_results(hash_t *hash, void *mem) {
  results_t *results = mem;

  __m256i nullBytes = _mm256_set1_epi8(0);

  // 0 is dummy
  results->num_results = hash->num_cities - 1;

  mem += sizeof(results_t) + results->num_results * sizeof(result_t *);

  for (int i = 1; i < hash->num_cities; i++) {
    result_t *r = mem;
    mem += sizeof(result_t);

    results->results[i - 1] = r;

    r->city = hash->packed_cities + i * SHORT_CITY_LENGTH;

    __m256i city = _mm256_load_si256((__m256i *)r->city);
    r->cityLength = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(city, nullBytes)));

    if (unlikely(r->cityLength == 0)) {
      int long_hash = _mm256_extract_epi32(city, 1);
      r->city = hash->hashed_cities_long + long_hash;

      __m256i seg0 = _mm256_load_si256((__m256i *)(r->city));
      __m256i seg1 = _mm256_load_si256((__m256i *)(r->city) + 1);
      __m256i seg2 = _mm256_load_si256((__m256i *)(r->city) + 2);
      __m256i seg3 = _mm256_load_si256((__m256i *)(r->city) + 3);

      r->cityLength  = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg0, nullBytes)));
      r->cityLength += _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg1, nullBytes)));
      r->cityLength += _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg2, nullBytes)));
      r->cityLength += _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(seg3, nullBytes)));
    }

    int hash_value = hash_city(city);
    int offset = get_offset(0, hash_value, city, hash);
    r->data = (result_data_t *)(hash->hashed_storage + offset);

    if (unlikely(_mm256_extract_epi32(city, 0) == 0)) {
      int long_hash = _mm256_extract_epi32(city, 1);
      r->city = hash->hashed_cities_long + long_hash;
    }
  }
  qsort(results->results, results->num_results, sizeof(result_t *), sort_result);
  return results;
}


int sort_result(const void *a, const void *b) {
  result_t *left  = *(result_t **)a;
  result_t *right = *(result_t **)b;
  return memcmp(left->city, right->city, LONG_CITY_LENGTH);
}

void print_results(results_t *results, void *mem) {
  char *buffer = mem;

  int pos = 0;
  buffer[pos++] = '{';

  for (int i = 0; i < results->num_results; i++) {
    result_t *r = results->results[i];
    memcpy(buffer + pos, r->city, r->cityLength);
    pos += r->cityLength;
    buffer[pos++] = '=';

    float sum = r->data->sum * 1.0;
    float count = r->data->count * 1.0;
    float min = r->data->min * 0.1;
    float max = r->data->max * 0.1;
    pos += sprintf(buffer + pos, "%.1f/%.1f/%.1f", min, round(sum/count) * 0.1, max);

    if (i != results->num_results - 1) {
      buffer[pos++] = ',';
      buffer[pos++] = ' ';
    }
  }
  buffer[pos++] = '}';
  buffer[pos++] = '\n';
  buffer[pos++] = '\0';
  fputs(buffer, stdout);
}

worker_t *get_worker(void *shared, int worker_id) {
  return (worker_t *)(shared + WORKER_MEMORY_SIZE * worker_id);
}

unsigned int find_next_row(const void *data, unsigned int offset) {
  __m256i newlines = _mm256_set1_epi8('\n');
  __m256i chars = _mm256_loadu_si256((__m256i *)(data + offset));
  unsigned int bytes = _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, newlines)));
  if (likely(bytes < 32)) {
    return offset + bytes + 1;
  }
  while (*((char *)data + offset + bytes) != '\n') {
    bytes++;
  }
  return offset + bytes + 1;
}

void debug_results(hash_t *hash) {
  char fullCity[LONG_CITY_LENGTH + 1] = {0};
  const char * dummyName = "__DUMMY__";

  fprintf(stderr, "\n");

  for (int i = 0; i < MIN(10, hash->num_cities); i++) {
    __m256i city = _mm256_load_si256((__m256i *)(hash->packed_cities + i * SHORT_CITY_LENGTH));
    if (i == 0) {
      strcpy(fullCity, dummyName);
    }
    else if (_mm256_extract_epi32(city, 0) != 0) {
      _mm256_storeu_si256((__m256i *)fullCity, city);
      fullCity[32] = 0;
    }
    else {
      int long_hash = _mm256_extract_epi32(city, 1);
      __m256i seg0 = _mm256_load_si256((__m256i *)(hash->hashed_cities_long + long_hash));
      __m256i seg1 = _mm256_load_si256((__m256i *)(hash->hashed_cities_long + long_hash) + 1);
      __m256i seg2 = _mm256_load_si256((__m256i *)(hash->hashed_cities_long + long_hash) + 2);
      __m256i seg3 = _mm256_load_si256((__m256i *)(hash->hashed_cities_long + long_hash) + 3);
      _mm256_storeu_si256((__m256i *)fullCity, seg0);
      _mm256_storeu_si256((__m256i *)fullCity + 1, seg1);
      _mm256_storeu_si256((__m256i *)fullCity + 2, seg2);
      _mm256_storeu_si256((__m256i *)fullCity + 3, seg3);
    }
    int h = hash_city(city);
    int offset = get_offset(0, h, city, hash);
    fprintf(stderr, "%-100s %12ld %11d %4d %4d\n",
      fullCity,
      *(long  *)(hash->hashed_storage + offset + 0),
      *(int   *)(hash->hashed_storage + offset + 8),
      *(short *)(hash->hashed_storage + offset + 12),
      *(short *)(hash->hashed_storage + offset + 14));

  }

	long total = 0;
	for (int i = 0; i < hash->num_cities; i++) {
    __m256i city = _mm256_load_si256((__m256i *)(hash->packed_cities + i * SHORT_CITY_LENGTH));
    int h = hash_city(city);
    int o = get_offset(0, h, city, hash);

	  total += *(int  *)(hash->hashed_storage + o + 8);
	}
  fprintf(stderr, "total: %ld\n", total);
}

void print256(__m256i var) {
    char val[32];
    memcpy(val, &var, sizeof(val));
    printf("%02x %02x %02x %02x  %02x %02x %02x %02x | %02x %02x %02x %02x  %02x %02x %02x %02x | %02x %02x %02x %02x  %02x %02x %02x %02x | %02x %02x %02x %02x  %02x %02x %02x %02x\n", 0xFF & val[0], 0xFF & val[1], 0xFF & val[2], 0xFF & val[3], 0xFF & val[4], 0xFF & val[5], 0xFF & val[6], 0xFF & val[7], 0xFF & val[8], 0xFF & val[9], 0xFF & val[10], 0xFF & val[11], 0xFF & val[12], 0xFF & val[13], 0xFF & val[14], 0xFF & val[15], 0xFF & val[16], 0xFF & val[17], 0xFF & val[18], 0xFF & val[19], 0xFF & val[20], 0xFF & val[21], 0xFF & val[22], 0xFF & val[23], 0xFF & val[24], 0xFF & val[25], 0xFF & val[26], 0xFF & val[27], 0xFF & val[28], 0xFF & val[29], 0xFF & val[30], 0xFF & val[31]);
}
