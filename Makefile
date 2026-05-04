CFLAGS ?= -Wall -Wextra -std=c11 -O2
CPPFLAGS ?= -Iinclude
LDFLAGS ?= -lm

LIB_SPMM_SRC := src/matmul/spmm_csr.c

.PHONY: all clean check

all: rv_sparse_tests axb_rowip

axb_rowip: src/matmul/AxBRowIP.c $(LIB_SPMM_SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ src/matmul/AxBRowIP.c $(LIB_SPMM_SRC) $(LDFLAGS)

rv_sparse_tests: tests/test_spmm_csr.c $(LIB_SPMM_SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_spmm_csr.c $(LIB_SPMM_SRC) $(LDFLAGS)

check: rv_sparse_tests
	./rv_sparse_tests

clean:
	rm -f axb_rowip rv_sparse_tests
