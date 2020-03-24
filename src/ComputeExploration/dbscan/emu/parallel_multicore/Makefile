#   make        -- compiles your project into program.exe
#   make clean  -- removes compiled item
#   make handin -- creates a project Zip file for hand in
#
# All .cpp flles are included.

#CC = g++
CC = emu-cc
SRCS = $(wildcard *.cpp)
HDRS = $(wildcard *.h)
OBJS = $(SRCS:.cpp=.o)
DIRS = $(subst /, ,$(CURDIR))
PROJ = omp_dbscan.mwx

APP = $(PROJ)
#CFLAGS= -c -fopenmp -O3
CFLAGS= -c
LDFLAGS=
#LIBS= -fopenmp -O3
LIBS = -lemu_c_utils

all: $(APP)

$(APP): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(APP) $(LIBS)

%.o: %.cpp $(HDRS) $(MF)
	$(CC) -S $<
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(APP)

