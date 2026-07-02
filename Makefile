# ==============================================================================
# rv-sparse Build System
# ==============================================================================

# NOTE:
# This Makefile builds and tests the complete public API path:
#
#   Public API
#   Dispatch layer
#   Internal wrappers
#   Raw pointer kernels
#
# If you want to test an isolated backend or experimental kernel directly,
# create a separate benchmark or test file instead of editing this Makefile.

# ------------------------------------------------------------------------------
# Build Configuration
# ------------------------------------------------------------------------------

TARGET_ARCH ?= native
BUILD_TYPE  ?= release

OPENMP   ?= 1
VALGRIND ?= 0

# ------------------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------------------

SRC_DIR     := src
INC_DIR     := include
OBJ_DIR     := obj
BIN_DIR     := bin
TEST_DIR    := tests
EXAMPLE_DIR := examples
LIB_DIR     := lib

TOOLS_DIR    := tools
TOOLS_SRC_DIR    := $(TOOLS_DIR)/src
TOOLS_INC_DIR    := $(TOOLS_DIR)/include

# ------------------------------------------------------------------------------
# Toolchain & Architecture Flags Selection
# ------------------------------------------------------------------------------

ifeq ($(TARGET_ARCH), riscv)
    CC := riscv64-linux-gnu-gcc
    AR := riscv64-linux-gnu-ar

    ARCH_FLAGS := -march=rv64gcv -mabi=lp64d -static
    EXEC_WRAPPER := qemu-riscv64-static -cpu rv64,v=true

    $(info ---> Target Architecture: RISC-V)
else
    CC := gcc
    AR := ar

    ifeq ($(VALGRIND),1)
        ARCH_FLAGS := -march=x86-64-v2
        $(info ---> Target Architecture: x86-64 (Valgrind))
    else
        ARCH_FLAGS := -march=native
        $(info ---> Target Architecture: Native Host)
    endif

    EXEC_WRAPPER :=
endif

# ------------------------------------------------------------------------------
# Compiler and Linker Flags
# ------------------------------------------------------------------------------

CFLAGS := -Wall -Wextra -Wpedantic -std=c11 -I$(INC_DIR) -I$(TOOLS_INC_DIR) $(ARCH_FLAGS)
LDFLAGS :=
LDLIBS := -lm

ifeq ($(BUILD_TYPE), debug)
    CFLAGS += -g -O0 -DDEBUG
    $(info ---> Build Type: DEBUG)
else
    CFLAGS += -O3 -flto
    LDFLAGS += -flto
    $(info ---> Build Type: RELEASE)
endif

# ------------------------------------------------------------------------------
# Source Files
# ------------------------------------------------------------------------------

# Exclude legacy standalone prototype because it may contain its own main().
EXCLUDED_SRCS := \
	$(SRC_DIR)/matmul/AxBRowIP.c

SRCS_ALL := $(shell find $(SRC_DIR) -name '*.c')
SRCS := $(filter-out $(EXCLUDED_SRCS),$(SRCS_ALL))

OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

STATIC_LIB := $(LIB_DIR)/librvsparse.a

# ------------------------------------------------------------------------------
# Tool Sources
# ------------------------------------------------------------------------------

TOOLS_SRCS := $(shell find $(TOOLS_SRC_DIR) -name '*.c')
TOOLS_OBJS := $(patsubst $(TOOLS_SRC_DIR)/%.c,$(OBJ_DIR)/tools/%.o,$(TOOLS_SRCS))

# ------------------------------------------------------------------------------
# Test Files
# ------------------------------------------------------------------------------

TEST_SRCS := $(shell find $(TEST_DIR) -name '*.c')
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BIN_DIR)/test/%,$(TEST_SRCS))

# ------------------------------------------------------------------------------
# Example Files
# ------------------------------------------------------------------------------

# Only build examples that are currently supported by the implemented API.
# Keep future or experimental examples out of this list until their backend exists.
EXAMPLE_SRCS := \
	$(EXAMPLE_DIR)/spgemm_csr_f32.c

EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.c,$(BIN_DIR)/examples/%,$(EXAMPLE_SRCS))

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

.PHONY: all dirs clean test check examples run-examples print-config

all: dirs $(STATIC_LIB) $(TOOLS_OBJS) $(TEST_BINS) $(EXAMPLE_BINS)

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)
	@mkdir -p $(BIN_DIR)/test
	@mkdir -p $(BIN_DIR)/examples
	@mkdir -p $(dir $(OBJS))
	@mkdir -p $(dir $(TOOLS_OBJS))

$(STATIC_LIB): $(OBJS)
	@echo "  AR      $@"
	@$(AR) rcs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/tools/%.o: $(TOOLS_SRC_DIR)/%.c | dirs
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/test/%: $(TEST_DIR)/%.c $(STATIC_LIB) $(TOOLS_OBJS) | dirs
	@echo "  CCLD    $@"
	@$(CC) $(CFLAGS) $< $(TOOLS_OBJS) -L$(LIB_DIR) -lrvsparse $(LDFLAGS) $(LDLIBS) -o $@

$(BIN_DIR)/examples/%: $(EXAMPLE_DIR)/%.c $(STATIC_LIB) $(TOOLS_OBJS) | dirs
	@echo "  CCLD    $@"
	@$(CC) $(CFLAGS) $< $(TOOLS_OBJS) -L$(LIB_DIR) -lrvsparse $(LDFLAGS) $(LDLIBS) -o $@

test: all
	@echo ""
	@echo "--- Running Tests ---"
	@for test_bin in $(TEST_BINS); do \
		echo "Executing $$test_bin..."; \
		$(EXEC_WRAPPER) ./$$test_bin || exit 1; \
	done
	@echo "--- All Tests Passed ---"
	@echo ""

check: test

examples: all

run-examples: all
	@echo ""
	@echo "--- Running Examples ---"
	@for example_bin in $(EXAMPLE_BINS); do \
		echo "Executing $$example_bin..."; \
		$(EXEC_WRAPPER) ./$$example_bin || exit 1; \
	done
	@echo "--- All Examples Finished ---"
	@echo ""

print-config:
	@echo "TARGET_ARCH  = $(TARGET_ARCH)"
	@echo "BUILD_TYPE   = $(BUILD_TYPE)"
	@echo "CC           = $(CC)"
	@echo "AR           = $(AR)"
	@echo "CFLAGS       = $(CFLAGS)"
	@echo "LDFLAGS      = $(LDFLAGS)"
	@echo "LDLIBS       = $(LDLIBS)"
	@echo "SRCS         = $(SRCS)"
	@echo "TOOLS_SRCS   = $(TOOLS_SRCS)"
	@echo "TOOLS_OBJS   = $(TOOLS_OBJS)"
	@echo "TEST_SRCS    = $(TEST_SRCS)"
	@echo "EXAMPLE_SRCS = $(EXAMPLE_SRCS)"

clean:
	@echo "  CLEAN   $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)"
	@rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)
