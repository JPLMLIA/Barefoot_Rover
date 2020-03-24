#include <mpi.h>

#include"RandomForest.h"
#include"MnistPreProcess.h"
#include <cmath>
#include "Globals.h"

/* ===== Data dependent constants ===== */
#if ALTERNATE_READ_FORMAT == 0
#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define FEATURE 784
#define NUMBER_OF_CLASSES 10
#endif
/* ==================================== */

int main(int argc, char * argv[])
{
	int rank, nproc;
	MPI_Status	status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	printf( "Node %d of %d reporting\n", rank, nproc );

	if(argc != 9) {
	  printf("Usage: \"./RandomForest <train_data> <train_labels> <test_data> <test_labels> <model_file> <num_trees> <tree_depth> <min_leaf_samples>\"\n");
	  exit(1);
	}

	int train_num;
	int test_num;
	int feature;

#if ALTERNATE_READ_FORMAT	
	readHeader(argv[1], &train_num, &feature);
	readHeader(argv[3], &test_num, &feature);
	feature -= 1; //Header includes dependent var
#else
	train_num = TRAIN_NUM;
	test_num = TEST_NUM;
	feature = FEATURE;
#endif

	printf("Test samples: %d\n", test_num);
	
    //1. prepare data
	
	float**trainset;
	float** testset;
	float*trainlabels;
	float*testlabels;
	trainset=new float*[train_num];
	testset=new float*[test_num];
	trainlabels=new float[train_num];
	testlabels=new float[test_num];
	for(int i=0;i<train_num;++i)
	{trainset[i]=new float[feature];}
	for(int i=0;i<test_num;++i)
	// TODO:  put all in one big TEST_NUM * FEATURE block
	{testset[i]=new float[feature];}
	
	if (rank == 0)
		{	// TODO:  just read data into node 0 at first, distribute later
		printf( "Starting test read\n" );
#if ALTERNATE_READ_FORMAT
		readData1(trainset,trainlabels,argv[1],argv[2]);
		readData1(testset,testlabels,argv[3],argv[4]);
#else		
		readData(trainset,trainlabels,argv[1],argv[2]);
		readData(testset,testlabels,argv[3],argv[4]);
#endif
		printf( "Finishing test read\n" );
		}
	MPI_Bcast(testlabels,test_num,MPI_FLOAT,0,MPI_COMM_WORLD);
	for(int i=0;i<test_num;++i)
		MPI_Bcast(testset[i],feature,MPI_FLOAT,0,MPI_COMM_WORLD);
//	readData(trainset,trainlabels,
//        "/Users/xinling/PycharmProjects/MNIST_data/train-images-idx3-ubyte",
//		"/Users/xinling/PycharmProjects/MNIST_data/train-labels-idx1-ubyte");
//	readData(testset,testlabels,
//        "/Users/xinling/PycharmProjects/MNIST_data/t10k-images-idx3-ubyte",
//		"/Users/xinling/PycharmProjects/MNIST_data/t10k-labels-idx1-ubyte");
    
    //2. create RandomForest class and set some parameters
	//	RandomForest randomForest(num_trees,max_depth,min_leaf_samples,0);	// the numbers are ignored
    
	//3. start to train RandomForest
//	randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,10,true,56);//regression
//	randomForest.train(trainset,trainlabels,train_num,feature,(int)sqrt(feature),false);//classification
	
    //restore model from file and save model to file
//	randomForest.saveModel("./RandomForest2.Model");
//	randomForest.readModel("./RandomForest2.Model", rank, nproc);

	int num_trees = atoi(argv[6]);
	int tree_depth = atoi(argv[7]);
	int min_leaf_samples = atoi(argv[8]);
	
	RandomForest randomForest(num_trees, tree_depth, min_leaf_samples,0);
	randomForest.readModel(argv[5], rank, nproc);
	
    //predict single sample
//  float resopnse;
//	randomForest.predict(testset[0],resopnse);
	
    //predict a list of samples
	float*responses=new float[test_num];
	printf( "proc %d: About to predict\n", rank);
	double start = MPI_Wtime();
	randomForest.predict(testset,test_num,responses,rank);
	double stop = MPI_Wtime();
	float errorRate=0;
	if (rank == 0)
	  {
	    for(int i=0;i<test_num;++i)
	      {
		if(responses[i]!=testlabels[i])
		  {
		    errorRate+=1.0f;
		  }
		//for regression
		//		float diff=abs(resopnses[i]-testlabels[i]);
		//		errorRate+=diff;
	      }
	    printf("the gross error rate is:%f\n", errorRate);
	    errorRate/=test_num;
	    printf("the total error rate is:%f\n", errorRate);
	    printf("Predict took: %f\n", stop - start);
	  }
	
	delete[] responses;
	for(int i=0;i<train_num;++i)
	  {delete[] trainset[i];}
	for(int i=0;i<test_num;++i)
	  {delete[] testset[i];}
	delete[] trainlabels;
	delete[] testlabels;
	delete[] trainset;
	delete[] testset;
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	exit(0);
	// The return, below, causes a crash when running under MPI; use exit()
	// return 0;
};
