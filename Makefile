SRC_DIR = src
OBJ_DIR = build
INC_DIR = include

# CC compiler options:
CC=nvcc
CC_FLAGS=-I$(INC_DIR)
CC_LIBS=

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-I$(INC_DIR)
NVCC_LIBS=

## Make variables ##

# Target executable name:
EXE = turbo

# Object files:
OBJS = $(OBJ_DIR)/turbo.o $(OBJ_DIR)/solver.o

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/* *.o $(OBJ_DIR)/$(EXE)
