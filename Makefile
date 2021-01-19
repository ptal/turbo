SRC_DIR = src
OBJ_DIR = build
INC_DIR = include

LIBXML2 = -lxml2 -I/usr/include/libxml2
LIBXCSP3 = -Llib/XCSP3-CPP-Parser/lib  -lxcsp3parser -Ilib/XCSP3-CPP-Parser/include
LIBS= $(LIBXML2) $(LIBXCSP3)

NVCC=nvcc
NVCC_FLAGS=-std=c++14 -g -arch=sm_70 -I/usr/local/cuda/include -I$(INC_DIR) 

EXE = turbo

# Object files:
SOURCES = $(SRC_DIR)/turbo.cu $(SRC_DIR)/solver.cu
INC_ONLY = $(INC_DIR)/XCSP3_turbo_callbacks.hpp $(INC_DIR)/vstore.cuh $(INC_DIR)/cuda_helper.hpp $(INC_DIR)/constraints.cuh $(INC_DIR)/model_builder.hpp

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(INC_ONLY) $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@ $(LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/* *.o $(OBJ_DIR)/$(EXE)
