clang-17 1brc.c -Wall -Werror -Wno-unused-parameter -std=c17 -march=native -mtune=native -Ofast -o 1brc

For optimal ~8 core results:
./1brc measurments.txt 8 1
