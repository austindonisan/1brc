clang-17 1brc.c -Wall -Werror -Wno-unused-parameter -march=native -mtune=native -std=c17 -o 1brc

For optimal ~8 core results:
./1brc measurments.txt 8 1
