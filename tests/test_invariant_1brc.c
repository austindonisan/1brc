#include <check.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Forward declaration of the vulnerable function from 1brc.c
extern void vulnerable_allocation(size_t count, size_t size);

START_TEST(test_allocation_overflow_invariant)
{
    // Invariant: Multiplication for allocation size must not overflow
    // or must be validated before allocation
    const struct {
        size_t count;
        size_t size;
        const char *description;
    } test_cases[] = {
        {SIZE_MAX, 2, "overflow to small size"},
        {SIZE_MAX / 2 + 1, 2, "boundary overflow"},
        {100, 16, "valid normal input"},
        {0, SIZE_MAX, "zero count with max size"},
        {SIZE_MAX, 1, "max count with unit size"}
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        // The test exercises the actual vulnerable function
        // The security property is that the program must not exhibit
        // undefined behavior due to allocation overflow
        vulnerable_allocation(test_cases[i].count, test_cases[i].size);
        
        // If we reach here without crashing, the test passes for this case
        // In a real scenario, we might add instrumentation to detect
        // if an undersized buffer was allocated
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_allocation_overflow_invariant);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}