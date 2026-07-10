# Build the wind-tunnel miniapp in this repository while linking against an
# existing MFEM build.  MFEM's NavierSolver is miniapp support code rather than
# part of libmfem, so it is compiled here from the selected MFEM source tree.

MFEM_DIR ?= ../../mfem
MFEM_BUILD_DIR ?= $(MFEM_DIR)
CONFIG_MK ?= $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

NAVIER_DIR ?= $(if $(wildcard $(MFEM_DIR)/miniapps/fluids/navier/navier_solver.cpp),$(MFEM_DIR)/miniapps/fluids/navier,$(MFEM_DIR)/miniapps/navier)
NAVIER_SOLVER_SRC = $(NAVIER_DIR)/navier_solver.cpp
TARGET = navier_windtunnel
OBJECTS = navier_windtunnel.o navier_solver_mfem.o

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(OBJECTS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(OBJECTS) -o $@ $(MFEM_LIBS)

navier_windtunnel.o: navier_windtunnel.cpp $(NAVIER_DIR)/navier_solver.hpp $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) -I$(NAVIER_DIR) -c $< -o $@

navier_solver_mfem.o: $(NAVIER_SOLVER_SRC) $(NAVIER_DIR)/navier_solver.hpp $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) -I$(NAVIER_DIR) -c $(NAVIER_SOLVER_SRC) -o $@

$(MFEM_LIB_FILE):
	$(error MFEM is not built. Set MFEM_DIR to its source tree and, if different, MFEM_BUILD_DIR to its build tree)

clean:
	rm -f $(TARGET) $(OBJECTS)
	rm -rf *.dSYM

help:
	@echo "Build: make -C mfem MFEM_DIR=/path/to/mfem"
	@echo "If MFEM was built out of source, also set MFEM_BUILD_DIR=/path/to/build."
