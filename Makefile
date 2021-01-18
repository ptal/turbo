SRC_DIR = src
OBJ_DIR = build
INC_DIR = include

LIBXML2 = -lxml2 -I/usr/include/libxml2
LIBXCSP3 = -Llib/XCSP3-CPP-Parser/lib  -lxcsp3parser -Ilib/XCSP3-CPP-Parser/include
LIBS= $(LIBXML2) $(LIBXCSP3)

CC=nvcc
NVCC=nvcc
CPP_FLAGS=-I$(INC_DIR) -std=c++17
NVCC_FLAGS=-arch=sm_75 $(CPP_FLAGS)


## Make variables ##

# Target executable name:
EXE = turbo

# Object files:
OBJS = $(OBJ_DIR)/turbo.o $(OBJ_DIR)/solver.o
INC_ONLY = $(INC_DIR)/XCSP3_turbo_callbacks.hpp $(INC_DIR)/vstore.cuh $(INC_DIR)/cuda_helper.hpp $(INC_DIR)/constraints.cuh

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(LIBS)

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp $(INC_ONLY)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh $(INC_ONLY)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/* *.o $(OBJ_DIR)/$(EXE)
