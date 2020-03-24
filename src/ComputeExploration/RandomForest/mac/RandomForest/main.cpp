#include"RandomForest.h"
#include"MnistPreProcess.h"
#include <iostream>

#define ONLY_CREATE_MODEL 1
#define ALTERNATE_READ_FORMAT 1

#if ALTERNATE_READ_FORMAT == 0

#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define FEATURE 784
#define NUMBER_OF_CLASSES 2

#endif

//Args: train data, train labels, test data, test labels, num_trees, max_depth, min_leaf_samples
int main(int argc, const char * argv[])
{
    //1. prepare data
	float**trainset;
	float** testset;
	float*trainlabels;
	float*testlabels;

	int num_trees = atoi(argv[5]);
	int max_depth = atoi(argv[6]);
	int min_leaf_samples = atoi(argv[7]);

	int train_num;
	int test_num;
	int feature;
	int number_of_classes;
	  
	std::cout << "About to read data" << std::endl;
	
#if ALTERNATE_READ_FORMAT
	if(argc != 8) {
	  std::cout << "Args required: <train data> <train labels> <test data> <test labels> <num_trees> <max_depth> <min_leaf_samples>" << std::endl;
	  exit(1);
	}
	
	readHeader(argv[1], &train_num, &feature);
	readHeader(argv[3], &test_num, &feature);
#else
	train_num = TRAIN_NUM;
	test_num = TEST_NUM;
	feature = FEATURE;
	number_of_classes = NUMBER_OF_CLASSES;
#endif
	
	std::cout << "Read header" << std::endl;

	std::cout << "train num = " << train_num << std::endl;
	std::cout << "test num = " << test_num << std::endl;
	std::cout << "feature = " << feature << std::endl;	

	trainset=new float*[train_num];
	testset=new float*[test_num];
	trainlabels=new float[train_num];
	testlabels=new float[test_num];
	for(int i=0;i<train_num;++i)
	{trainset[i]=new float[feature];}
	for(int i=0;i<test_num;++i)
	{testset[i]=new float[feature];}

	std::cout << "Allocated space" << std::endl;
	
#if ALTERNATE_READ_FORMAT
	readData1(trainset, trainlabels, argv[1], argv[2], &number_of_classes);
	int throw_away;
	readData1(testset, testlabels, argv[3], argv[4], &throw_away);
#else
	readData(trainset,trainlabels,argv[1],argv[2]);
	readData(testset,testlabels,argv[3],argv[4]);
#endif
	
	std::cout << "Read data" << std::endl;
	std::cout << "Number of classes = " << number_of_classes << std::endl;
	
//	readData(trainset,trainlabels,
//        "/Users/xinling/PycharmProjects/MNIST_data/train-images-idx3-ubyte",
//		"/Users/xinling/PycharmProjects/MNIST_data/train-labels-idx1-ubyte");
//	readData(testset,testlabels,
//        "/Users/xinling/PycharmProjects/MNIST_data/t10k-images-idx3-ubyte",
//		"/Users/xinling/PycharmProjects/MNIST_data/t10k-labels-idx1-ubyte");
    
    //2. create RandomForest class and set some parameters
    RandomForest randomForest(num_trees,max_depth,min_leaf_samples,0);
    //RandomForest randomForest(100,10,10,0);


    
	//3. start to train RandomForest
//	randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,10,true,56);//regression
    randomForest.train(trainset,trainlabels,train_num,feature,number_of_classes,false);//classification

#if ONLY_CREATE_MODEL
    
    //restore model from file and save model to file
    char fn[500];
    sprintf(fn, "%d_%d_%d.Model", num_trees, max_depth, min_leaf_samples);
    randomForest.saveModel(fn);
    return 0;
//	randomForest.readModel("./RandomForest2.Model");
//	RandomForest randomForest("E:\\RandomForest2.Model");

#endif
    
    //predict single sample
//  float resopnse;
//	randomForest.predict(testset[0],resopnse);
    
    //predict a list of samples
    float*resopnses=new float[test_num];
	randomForest.predict(testset,test_num,resopnses);
	float errorRate=0;
	for(int i=0;i<test_num;++i)
	{
        if(resopnses[i]!=testlabels[i])
        {
            errorRate+=1.0f;
        }
        //for regression
//		float diff=abs(resopnses[i]-testlabels[i]);
//		errorRate+=diff;
	}
	errorRate/=test_num;
	printf("the total error rate is:%f\n",errorRate);

	delete[] resopnses;
	for(int i=0;i<train_num;++i)
	{delete[] trainset[i];}
	for(int i=0;i<test_num;++i)
	{delete[] testset[i];}
	delete[] trainlabels;
	delete[] testlabels;
	delete[] trainset;
	delete[] testset;
	return 0;
};
