CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=c11
INCLUDES = -Iinclude

BUILD_DIR = build

SRC = \
	src/core/matrix.c \
	src/core/error.c \
	src/core/spgemm.c \
	src/kernel/spgemm/csr_spgemm_wrappers.c\
	src/kernels/spgemm/csr_scalar_f32.c

EXAMPLE_F32 = $(BUILD_DIR)/spgemm_csr_f32
TEST_F32 = $(BUILD_DIR)/test_spgemm_csr_f32

.PHONY: all run test check clean

all: $(EXAMPLE_F32) $(TEST_F32)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXAMPLE_F32): $(SRC) examples/spgemm_csr_f32.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) examples/spgemm_csr_f32.c -o $(EXAMPLE_F32)

$(TEST_F32): $(SRC) tests/test_spgemm_csr_f32.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) tests/test_spgemm_csr_f32.c -o $(TEST_F32)

run: $(EXAMPLE_F32)
	./$(EXAMPLE_F32)

test: $(TEST_F32)
	./$(TEST_F32)

check: all test

clean:
	rm -rf $(BUILD_DIR)
