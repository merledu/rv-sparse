# Makefile
CC_NATIVE = gcc
CC_RISCV  = riscv64-linux-gnu-gcc
QEMU      = qemu-riscv64-static

CFLAGS_COMMON = -Wall -Wextra -I include
FLAGS_NATIVE = $(CFLAGS_COMMON) -O2 -lm
CFLAGS_RISCV  = $(CFLAGS_COMMON) -O2 -march=rv64gcv -mabi=lp64d -static -lm 

SRC_CORE = src/matmul/csr.c src/matmul/gen.c src/matmul/spmv_scalar.c src/matmul/spmm_scalar.c

# Native targets (for fast iteration)
test_native: $(SRC_CORE) src/matmul/test_correctness.c
	$(CC_NATIVE) $(CFLAGS_NATIVE) -o $@ $^

# RISC-V targets
test_riscv: $(SRC_CORE) src/matmul/spmv_rvv.c src/matmul/spmm_rvv.c src/matmul/test_correctness.c
	$(CC_RISCV) $(CFLAGS_RISCV) -o $@ $^

test_rvv_ops: src/matmul/test_rvv_ops.c
	$(CC_RISCV) $(CFLAGS_RISCV) -o $@ $^

run_rvv_ops: test_rvv_ops
	$(QEMU) -cpu rv64,v=true,vlen=256 ./test_rvv_ops

bench_riscv: $(SRC_CORE) src/matmul/spmv_rvv.c src/matmul/spmm_rvv.c src/matmul/benchmark.c
	$(CC_RISCV) $(CFLAGS_RISCV) -o $@ $^

bench_spmm_riscv: $(SRC_CORE) src/matmul/spmm_rvv.c src/matmul/benchmark_spmm.c
	$(CC_RISCV) $(CFLAGS_RISCV) -o $@ $^


run_test: test_native test_riscv test_rvv_ops
	./test_native
	$(QEMU) -cpu rv64,v=true,vlen=256 ./test_riscv
	$(QEMU) -cpu rv64,v=true,vlen=256 ./test_rvv_ops

run_bench: bench_riscv bench_spmm_riscv
	$(QEMU) -cpu rv64,v=true,vlen=256 ./bench_riscv
	$(QEMU) -cpu rv64,v=true,vlen=256 ./bench_spmm_riscv

run_bench_spmm: bench_spmm_riscv
	$(QEMU) -cpu rv64,v=true,vlen=256 ./bench_spmm_riscv


clean:
	rm -f test_native test_riscv test_rvv test_rvv_ops bench_riscv bench_spmm_riscv a.out run.exe
