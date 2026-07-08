CC = gcc
CFLAGS = -Wall -Wextra -O3 -Iinclude

SRCS = src/csr.c src/io.c src/spmv.c src/spgemm.c
OBJS = $(SRCS:.c=.o)

TEST_SPMV = test/test_spmv
TEST_SPGEMM = test/test_spgemm
BENCH_SPGEMM = bench/bench_spgemm

all: $(TEST_SPMV) $(TEST_SPGEMM) $(BENCH_SPGEMM)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_SPMV): test/test_spmv.c $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

$(TEST_SPGEMM): test/test_spgemm.c $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

$(BENCH_SPGEMM): bench/bench_spgemm.c $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

check: $(TEST_SPMV) $(TEST_SPGEMM)
	@echo "Running tests..."
	@./$(TEST_SPMV)
	@./$(TEST_SPGEMM)
	@echo "All tests passed!"

bench: $(BENCH_SPGEMM)
	@echo "Running benchmarks..."
	@./$(BENCH_SPGEMM) 500 0.05
	@./$(BENCH_SPGEMM) 1000 0.01

clean:
	rm -f src/*.o test/*.o bench/*.o $(TEST_SPMV) $(TEST_SPGEMM) $(BENCH_SPGEMM)

.PHONY: all check bench clean
