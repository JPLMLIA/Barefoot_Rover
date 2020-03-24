default: random_forest_classifier.mwx

random_forest_classifier.o: random_forest_classifier.cpp random_forest_classifier.h training.h
	emu-cc -c random_forest_classifier.cpp -I.

test.o: test.c
	emu-cc -c test.c -I.

random_forest_classifier.mwx: random_forest_classifier.o test.o
	emu-cc -o random_forest_classifier.mwx random_forest_classifier.o test.o

clean:
	rm -f random_forest_classifier *.mwx *.cdc *.tqd *.vsf *.o

