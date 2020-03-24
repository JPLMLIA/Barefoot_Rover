#include <iostream>
#include <vector>
#include"MnistPreProcess.h"

void readData(float** dataset,float*labels,const char* dataPath,const char*labelPath)
{
	FILE* dataFile=fopen(dataPath,"rb");
	FILE* labelFile=fopen(labelPath,"rb");
	int mbs=0,number=0,col=0,row=0;
	fread(&mbs,4,1,dataFile);
	fread(&number,4,1,dataFile);
	fread(&row,4,1,dataFile);
	fread(&col,4,1,dataFile);
	revertInt(mbs);
	revertInt(number);
	revertInt(row);
	revertInt(col);
	fread(&mbs,4,1,labelFile);
	fread(&number,4,1,labelFile);
	revertInt(mbs);
	revertInt(number);
	unsigned char temp;
	for(int i=0;i<number;++i)
	{
		for(int j=0;j<row*col;++j)
		{
			fread(&temp,1,1,dataFile);
			dataset[i][j]=static_cast<float>(temp);
		}
		fread(&temp,1,1,labelFile);
		labels[i]=static_cast<float>(temp);
	}
	fclose(dataFile);
	fclose(labelFile);
};

void readData1(float** dataset,float*labels,const char* dataPath,const char*labelPath, int * outNumClasses)
{
  FILE* dataFile=fopen(dataPath,"rb");
  FILE* labelFile=fopen(labelPath,"rb");
  int n;
  int d;
  fread(&n,4,1,dataFile);
  fread(&d,4,1,dataFile);	
  float temp;
  std::cout << "n = "<< n << std::endl;
  std::cout << "d = " << d << std::endl;	
  std::vector<float> ulbls = std::vector<float>();
  for(int i=0;i<n;++i)
    {
      for(int j=0;j<d-1;++j)
	{
	  fread(&temp,4,1,dataFile);
	  dataset[i][j] = temp;
	}
      fread(&temp,4,1,labelFile);
      labels[i] = temp;
      bool found = false;
      for(int i = 0; i < ulbls.size(); ++i) {
	if(ulbls[i] == temp) {
	  found = true;
	  break;
	}
      }
      if(!found) {
	ulbls.push_back(temp); //keep track of unique labels to count num classes
      }
    }
  fclose(dataFile);
  fclose(labelFile);
  *outNumClasses = ulbls.size();
};

void readHeader(const char* dataPath, int * n, int * d) {
  	FILE* dataFile=fopen(dataPath,"rb");
	fread(n,4,1,dataFile);
	fread(d,4,1,dataFile);
	fclose(dataFile);	
}
