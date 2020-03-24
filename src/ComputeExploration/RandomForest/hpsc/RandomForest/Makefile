#CC     = gcc
#CFLAGS = -Wall
CFLAGS = -Wall

#include messaging/mw_messaging_files.inc

.PHONY: FORCE

all: RandomForest

RandomForest: FORCE
	$(CXX) $(CFLAGS) -Iinclude $(MW_MESSAGING_FILES) \
		main.cpp MnistPreProcess.cpp Node.cpp RandomForest.cpp Sample.cpp Tree.cpp -lmpi -lc -lm -o RandomForest

clean:
	rm -f RandomForest
