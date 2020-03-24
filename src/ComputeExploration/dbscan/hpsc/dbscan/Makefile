#CC     = gcc
#CFLAGS = -Wall
CFLAGS = -Wall -std=c99

#include messaging/mw_messaging_files.inc

.PHONY: FORCE

all: dbscan

dbscan: FORCE
	$(CC) $(CFLAGS) -Iinclude $(MW_MESSAGING_FILES) \
		dbscan.c -lc -lm -o dbscan

clean:
	rm -f dbscan
