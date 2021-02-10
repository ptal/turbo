SRC_DIR = src
OBJ_DIR = build
INC_DIR = include

LIBXML2 = -lxml2 -I/usr/include/libxml2
LIBXCSP3 = -Llib/XCSP3-CPP-Parser/lib  -lxcsp3parser -Ilib/XCSP3-CPP-Parser/include
LIBS= $(LIBXML2) $(LIBXCSP3)

NVCC=nvcc
NVCC_FLAGS=-std=c++14 -rdc=true -arch=sm_75 -I/usr/local/cuda/include -I$(INC_DIR)

EXE = turbo
EXE_SEQ = turbo_seq

# Object files:
SOURCES = $(SRC_DIR)/turbo.cu $(SRC_DIR)/solver.cu
INC_ONLY = $(INC_DIR)/*

## Compile ##

debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(EXE)

trace: NVCC_FLAGS += -DTRACE
trace: $(EXE)

compete: NVCC_FLAGS += -O3
compete: $(EXE)

$(EXE): $(INC_ONLY) $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@ $(LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(EXE) $(EXE_SEQ)
