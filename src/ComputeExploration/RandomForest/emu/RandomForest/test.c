//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: test.c
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Random Forest Classifier: independent test
// ========================================================================

#include <float.h>
//#include <limits>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//#include <sstream>
//#include <vector>

#include <cilk.h>
#include <memoryweb.h>

#include "rfc_api.h"
#include "log.h"
//#include "training.h"
#include "globals.h"

void readData(const char*dataPath, const char *path2,uint64_t *numFeatures, uint64_t *numTestSamples, double **testSamples, uint64_t **testLabels);
void readData1(const char*dataPath, const char *labelPath, double **testSamples, uint64_t **testLabels);
void readModel(const char*path, uint64_t* numClasses);
void parseFile_sklearn(const char *filename, double trainPercent, uint64_t *numFeatures, uint64_t *numClasses, 
		uint64_t *numTrainSamples, double **trainSamples, uint64_t **trainLabels, 
		uint64_t *numTestSamples, double **testSamples, uint64_t **testLabels);
void readHeader(const char*dataPath, uint64_t * n, uint64_t * d);


inline void revertInt(int *x)
{
	*x=((*x&0x000000ff)<<24)|((*x&0x0000ff00)<<8)|((*x&0x00ff0000)>>8)|((*x&0xff000000)>>24);
};

int main(int argc, char **argv) {
	char filename[4096];
	uint64_t	testSampleCount;
	uint64_t	testTreeCount;
	uint64_t	threadsPerSample;
	uint64_t	treesPerThread;

	uint64_t numTrainSamples;
	double *trainSamples;
	uint64_t *trainLabels;
	uint64_t numTestSamples;
	double *testSamples;
	uint64_t *testLabels;
	uint64_t numFeatures;
	uint64_t numClasses;
	
#if ALTERNATE_READ_FORMAT
	if(argc != 10) {
          fprintf(stderr, "Argument needed: <train_data> <train_labels> <test_data> <test_labels> <model_file> <num_trees> <tree_depth> <max_test_samples> <treesPerThread>");
        }

	readHeader(argv[1], &numTrainSamples, &numFeatures);
	readHeader(argv[3], &numTestSamples, &numFeatures);

	//LOG_DEBUG("Test samples: %d\n", numTestSamples);
	
	/*trainSamples = malloc(numTrainSamples * sizeof(double));
	trainLabels = malloc(numTrainSamples * sizeof(uint64_t));
	testSamples = malloc(numTestSamples * sizeof(double));
	testLabels = malloc(numTestSamples * sizeof(uint64_t));*/
	
	readData1(argv[1], argv[2], &trainSamples, &trainLabels);
	readData1(argv[3], argv[4], &testSamples, &testLabels);

	//Initialize trees
	randomForestClassifier_RMinitialize(atol(argv[6]), atol(argv[6]), atol(argv[9]));

	//Read in model
	readModel(argv[5], &numClasses);
	
	//Train
	//numClasses = 2;
	//randomForestClassifier_initializeTraining(numTrainSamples, numFeatures, numClasses);
	//randomForestClassifier_fillTrainingData(trainSamples, trainLabels);
	//randomForestClassifier_train();

	testSampleCount = atoi(argv[8]);	// limit sample count
	if(testSampleCount <= 0 || testSampleCount > numTestSamples) {
	  testSampleCount = numTestSamples;
	}
	
#else /////////////////////////////////////////////////////////////////////////////
	if (argc < 7) {
		fprintf(stderr, "Argument needed: <data file name> <num_trees> <selection_size> <training ratio> <max sample count> <max tree count> [<threads per sample>]\n");
		return -1;
	}
	strcpy(filename, argv[1]);
	//mw_replicated_init((long *)&g_numTrees, atol(argv[2]));
	//mw_replicated_init((long *)&g_selectionSize, atol(argv[3]));

	//randomForestClassifier_initialize(atol(argv[2]), AUTO, 0);

	// set up the trees for reading the model from file
	// which already has 100 trees in it
	testTreeCount = atol(argv[6]);	// limit tree count
	threadsPerSample = testTreeCount;	// default:  1 tree per thread
	if (argc == 8)
		{
		threadsPerSample = atol(argv[7]);	// threads per sample
		// vary trees per thread to implement threads per sample
		// for now user must make sure this is evenly divisible
		treesPerThread = testTreeCount / threadsPerSample;
		}
	else
		treesPerThread = 1;
	LOG_DEBUG( "Running on %d nodelets\n", NODELETS() );
	LOG_DEBUG( "Requesting %d threads per sample\n", threadsPerSample );
	randomForestClassifier_RMinitialize(100, testTreeCount, treesPerThread);
	
	double trainPercent = atof(argv[4]);
	
	readData("./t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &numFeatures, &numTestSamples, &testSamples, &testLabels);
	LOG_DEBUG( "Test data read in\n" );
	readModel("./RandomForest2.Model", &numClasses);
	LOG_DEBUG( "Model read in\n" );
	testSampleCount = atoi(argv[5]);	// limit sample count

#endif ///////////////////////////////////////////////////////////////////////

	/*bool cycle_accurate = false; //TODO kills runtime
	  if(cycle_accurate) {
	  starttiming();
	  }*/

	randomForestClassifier_RMinitializeTesting(numClasses,numFeatures,numTestSamples,testSampleCount);
	randomForestClassifier_RMfillTestingData(testSamples);
	uint64_t *classified = (uint64_t *)malloc(numTestSamples * sizeof(uint64_t));
	LOG_DEBUG( "About to start testing\n" );
	volatile uint64_t starttimer = CLOCK();
	randomForestClassifier_test(classified);

	uint64_t right = 0;
	uint64_t wrong = 0;

	for (uint64_t i = 0; i < testSampleCount; i++) {
		if (i < 10)
			LOG_DEBUG("[%lu] %lu / %lu\n", i, classified[i], testLabels[i]);
		if (classified[i] == testLabels[i])
			right++;
		else
			wrong++;
	}

	//free(testSamples);
	//free(testLabels);
	free(classified);

	volatile uint64_t endtimer = CLOCK();
	uint64_t delta = endtimer - starttimer;
	LOG_DEBUG("Completed in (cycles): %lu\n", delta);
	//LOG_INFO("Completed in %lu cycles Right %lu vs Wrong %lu (accuracy %.2f%%)\n", delta, right, wrong, (double)right / (double)(right + wrong) * 100.0);
	LOG_INFO("Completed  Right %lu vs Wrong %lu (error rate %.4f)\n", right, wrong, (double)wrong / (double)(right + wrong));

	return 0;

#if 0
	parseFile_sklearn(filename, trainPercent, &numFeatures, &numClasses, &numTrainSamples, &trainSamples, &trainLabels, &numTestSamples, &testSamples, &testLabels);

	LOG_INFO("=== Reading %s #samples: %lu %lu #features: %lu #classes: %lu ===\n", filename, numTrainSamples, numTestSamples, numFeatures, numClasses);

	randomForestClassifier_initializeTraining(numTrainSamples, numFeatures, numClasses);
	randomForestClassifier_fillTrainingData(trainSamples, trainLabels);
	free(trainSamples);
	free(trainLabels);

	starttiming();
	volatile uint64_t starttimer = CLOCK();
	randomForestClassifier_train();

	randomForestClassifier_initializeTesting(numTestSamples);
	randomForestClassifier_fillTestingData(testSamples); //, testLabels);
	uint64_t *classified = (uint64_t *)malloc(numTestSamples * sizeof(uint64_t));
	randomForestClassifier_test(classified);

	uint64_t right = 0;
	uint64_t wrong = 0;

	for (uint64_t i = 0; i < numTestSamples; i++) {
		//LOG_DEBUG("[%lu] %lu / %lu\n", i, classified[i], testLabels[i]);
		if (classified[i] == testLabels[i])
			right++;
		else
			wrong++;
	}

	//free(testSamples);
	//free(testLabels);
	free(classified);

	volatile uint64_t endtimer = CLOCK();
	uint64_t delta = endtimer - starttimer;
	//LOG_INFO("Completed in %lu cycles\n", delta);
	LOG_INFO("Completed in %lu cycles Right %lu vs Wrong %lu (accuracy %.2f%%)\n", delta, right, wrong, (double)right / (double)(right + wrong) * 100.0);
#endif

	return 0;
}

// Assuming the following format:
// First line: number of entries,number of features per line
// All others: CSV features. We assume only the useful ones have already been selected
#if 0
void parseFile_sklearn(const char *filename, double trainPercent, uint64_t *numFeatures, uint64_t *numClasses, uint64_t *numTrainSamples, double **trainSamples, uint64_t **trainLabels, uint64_t *numTestSamples, double **testSamples, uint64_t **testLabels) {
	FILE *fd = fopen(filename, "r");
	if (!fd) {
		fprintf(stderr, "Could not open data file %s\n", filename);
		exit(-1);
	}
	uint64_t nsamp, nf;
	fscanf(fd, "%lu,%lu,", &nsamp, &nf);
	*numFeatures=  nf;

	uint64_t foldValue = 1 / (1.0 - trainPercent);
	LOG_DEBUG("folded: %lu\n", foldValue);

	*numTestSamples = nsamp / foldValue;
	*numTrainSamples = nsamp - *numTestSamples;

	uint64_t nc = 1;
	while (true) {
		char c;
		fread(&c, 1, sizeof(char), fd);
		if (c == ',')
			nc++;
		if (c == '\n')
			break;
	}
	*numClasses = nc;

	*trainSamples = (double *)malloc(*numTrainSamples * nf * sizeof(double));
	*trainLabels = (uint64_t *)malloc(*numTrainSamples * sizeof(uint64_t));
	double *gc_trainSamples = *trainSamples;
	uint64_t *gc_trainLabels = *trainLabels;

	*testSamples = (double *)malloc(*numTestSamples * nf * sizeof(double));
	uint64_t *gc_testLabels = (uint64_t *)malloc(*numTestSamples * sizeof(uint64_t));
	double *gc_testSamples = *testSamples;
	//uint64_t *gc_testLabels = *testLabels;

	uint64_t ntr = 0; // training
	uint64_t ntt = 0; // test
	for (uint64_t s = 0; s < nsamp; s++) {
		bool testing = (ntt < *numTestSamples ? ((s+1) % foldValue == 0) :  false);
		for (uint64_t f = 0; f < nf; f++) {
			double val;
			fscanf(fd, "%lf,", &val);
			if (!testing) {
				gc_trainSamples[ntr * nf + f] = val;
	//			g_trainSamples[ntr][f] = val;
			} else {
#ifdef INVERT_TEST_DATA
				gc_testSamples[f * (*numTestSamples) + ntt] = val;
				//g_testSamples[f][ntt] = val;
#else
				gc_testSamples[ntt * nf + f] = val;
				//g_testSamples[ntt][f] = val;
#endif
			}
		}
		uint64_t label;
		fscanf(fd, "%lu\n", &label);
		if (!testing) {
			gc_trainLabels[ntr] = label;
	//		g_trainSamplesLabels[ntr] = label;
			ntr++;
		} else {
			gc_testLabels[ntt] = label;
			//g_testSamplesLabels[ntt++] = label;
			ntt++;
		}
	}
	fclose(fd);

	*testLabels = gc_testLabels;
}
#endif

void readData(const char* dataPath, const char* labelPath, uint64_t *numFeatures, uint64_t *numTestSamples, double **testSamples, uint64_t **testLabels)
	// read in the test data file
	{
	FILE* dataFile=fopen(dataPath,"rb");
	FILE* labelFile=fopen(labelPath,"rb");
	int mbs=0,number=0,col=0,row=0;
	fread(&mbs,4,1,dataFile);
	fread(&number,4,1,dataFile);
	fread(&row,4,1,dataFile);
	fread(&col,4,1,dataFile);
	revertInt(&mbs);
	revertInt(&number);
	revertInt(&row);
	revertInt(&col);
	
	fread(&mbs,4,1,labelFile);
	fread(&number,4,1,labelFile);
	revertInt(&mbs);
	revertInt(&number);

	*numTestSamples = number;
	*numFeatures = row * col;
	printf( "mbs is %d\n", mbs );
	printf( "number is %d\n", *numTestSamples );
	printf( "row is %d\n", row );
	printf( "col is %d\n", col );
	printf( "numFeatures is %d\n", *numFeatures );

	unsigned char	temp;
	*testSamples = (double *)malloc(number * row * col * sizeof(double));
	uint64_t *gc_testLabels = (uint64_t *)malloc(*numTestSamples * sizeof(uint64_t));
	double *gc_testSamples = *testSamples;

	for(int i=0;i<number;++i)
		{
		//printf( "Reading vector %d\n", i );
		for(int j=0;j<row*col;++j)
			{	// cycle through features
			fread(&temp,1,1,dataFile);
			gc_testSamples[i*row*col + j] = (double)(temp);
			}
		fread(&temp,1,1,labelFile);
		gc_testLabels[i] = (uint64_t)(temp);
		if (i < 5)
			printf( "Test vector %d has label %d\n", i, gc_testLabels[i] );
		}
	fclose(dataFile);
	fclose(labelFile);

	*testLabels = gc_testLabels;
	}	// readData()

void readData1(const char* dataPath, const char* labelPath, double **testSamples, uint64_t **testLabels)
{
  //HPSC implementation uses float type for data, keeping consistent
  FILE* dataFile = fopen(dataPath, "rb");
  FILE* labelFile = fopen(labelPath, "rb");
  int n = 0, d = 0;
  fread(&n, 4, 1, dataFile);
  fread(&d, 4, 1, dataFile);
  float temp;
  *testLabels = (uint64_t *)malloc(n * sizeof(uint64_t));
  *testSamples = (double *)malloc(n * (d-1) * sizeof(double));
  uint64_t *gc_testLabels = *testLabels;
  double *gc_testSamples = *testSamples;
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < d-1; ++j) {
      fread(&temp, 4, 1, dataFile);
      gc_testSamples[i * (d-1) + j] = temp;
    }
    fread(&temp, 4, 1, dataFile);
    gc_testLabels[i] = (uint64_t)temp;
  }

  fclose(dataFile);
  fclose(labelFile);
}

void readHeader(const char*dataPath, uint64_t * n, uint64_t * d)
{
  FILE* dataFile = fopen(dataPath, "rb");
  int n32;
  int d32;
  fread(&n32, 4, 1, dataFile);
  fread(&d32, 4, 1, dataFile);
  *n = n32;
  *d = d32;
  fclose(dataFile);
};

void readModel(const char*path, uint64_t *numClasses)
	{
	int		_treeNum;
	int		_maxDepth;
	int		_classNum;
	bool	_isRegression;

	FILE* modelFile=fopen(path,"rb");
	fread(&_treeNum,sizeof(int),1,modelFile);
	fread(&_maxDepth,sizeof(int),1,modelFile);
	fread(&_classNum,sizeof(int),1,modelFile);
	fread(&_isRegression,sizeof(bool),1,modelFile);
	int nodeNum=(int)(pow(2.0,_maxDepth)-1);
	// _trainSample=NULL;
	//printf("File tree count of %d now set to 100\n", _treeNum);
	//_treeNum = 100;
	*numClasses = _classNum;
	printf("total tree number:%d\n",_treeNum);
	printf("max depth of a single tree:%d\n",_maxDepth);
	printf("_classNum:%d\n",_classNum);
	printf("_isRegression:%d\n",_isRegression);
	printf("nodeNum:%d\n",nodeNum);

	for(int i=0;i<_treeNum;++i)
		{
		randomForestClassifier_readModel(i, nodeNum, modelFile);
		//return;
		}
	}
