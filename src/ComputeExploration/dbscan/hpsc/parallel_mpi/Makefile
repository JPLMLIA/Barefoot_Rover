#   make        -- compiles your project into program.exe
#   make clean  -- removes compiled item
#   make handin -- creates a project Zip file for hand in
#
# All .cpp flles are included.

#CC = mpicc
SRCS = $(wildcard *.cpp)
HDRS = $(wildcard *.h)
OBJS = $(SRCS:.cpp=.o)
DIRS = $(subst /, ,$(CURDIR))
PROJ = mpi_dbscan

APP = $(PROJ)
CFLAGS= -c -O3
LDFLAGS=
#LIBS= -lstdc++ -O3 -lpnetcdf
LIBS= -lmpi -lstdc++ -O3

all: $(APP)

$(APP): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $(APP) $(LIBS)

%.o: %.cpp $(HDRS) $(MF)
	$(CXX) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(APP)

