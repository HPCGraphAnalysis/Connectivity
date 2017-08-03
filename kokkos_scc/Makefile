KOKKOS_PATH = /lore/cwsmith/develop/build-jessie-kokkos/install
#SRC = $(wildcard *.cpp)
SRC = scc_main.cpp

ifeq ($(CUDA), yes)
CC = nvcc
CCFLAGS = -arch=sm_35 -O3
LINK = g++
LINKFLAGS = -L/home/projects/cuda/5.5.11/lib64 -lcudart -fopenmp
EXE = $(SRC:.cpp=.cuda)
else
CC = g++
CCFLAGS = -fopenmp -O3
LINK = g++
LINKFLAGS = -fopenmp
EXE = $(SRC:.cpp=.host)
endif

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

EXTRA_FLAGS = $(KOKKOS_CXXFLAGS)
EXTRA_INC = $(KOKKOS_CPPFLAGS)
EXTRA_LIB = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) 

all: $(OBJ)
	$(LINK) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(EXTRA_LIB) -o $(EXE)

clean: 
	rm *.o *.d *.a

# Compilation rules

%.o:%.cu
	$(CC) $(CCFLAGS) $(EXTRA_INC) -c $<

%.o:%.cpp
	$(CC) $(CCFLAGS) $(EXTRA_FLAGS) $(EXTRA_INC) -c $<

%.d:%.cpp
	$(CC) $(CCFLAGS) $(EXTRA_FLAGS) $(EXTRA_INC) $(DEPFLAGS) $< > $@

# Individual dependencies

DEPENDS = $(OBJ:.o=.d)
sinclude $(DEPENDS)

