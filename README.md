#### Buiding
`clang-17 1brc.c -std=c17 -march=native -mtune=native -Ofast -o 1brc`

AVX2 is required. GCC also works, but its binary is ~15% slower.

#### Running
`./1brc measurments.txt n`

_n_ is the number of threads to use

Work is spread evenly across threads with no rebalancing. If using hypethreading, _n_ must equal the number of available logical cores for best performance. Use `taskset` to limit the available logical cores.

#### Implementation summary
[Fastest known solution: 0.577s (8 core Zen2); C with heavy AVX2](https://github.com/gunnarmorling/1brc/discussions/710)
