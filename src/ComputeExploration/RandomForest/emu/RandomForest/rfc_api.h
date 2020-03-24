//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: rfc_api.h
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Random Forest Classifier API: GC side!
// ========================================================================
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// At each split, how many features do we choose from?
enum MaxFeatures {
	AUTO,
	SQRT,
	LOG2,
	NONE
};

void randomForestClassifier_initialize(uint64_t numTrees, enum MaxFeatures maxFeatures, uint64_t randomState);
void randomForestClassifier_RMinitialize(uint64_t numTrees, uint64_t treeMax, uint64_t threadsPerTree);

// Train / fit
void randomForestClassifier_initializeTraining(uint64_t numSamples, uint64_t numFeatures, uint64_t numClasses);
void randomForestClassifier_fillTrainingData(double *gc_samples, uint64_t *gc_classes);
void randomForestClassifier_train();
void randomForestClassifier_readModel(uint64_t t, int nodeNum, FILE* modelFile);

// Test / Predict
void randomForestClassifier_initializeTesting(uint64_t numSamples);
void randomForestClassifier_RMinitializeTesting(uint64_t numClasses, uint64_t numFeatures, uint64_t numSamples, uint64_t testSampleMax);
void randomForestClassifier_fillTestingData(double *gc_samples); //, uint64_t *gc_classes);
void randomForestClassifier_RMfillTestingData(double *gc_samples);
void randomForestClassifier_test(uint64_t *gc_classified);

#ifdef __cplusplus
}
#endif
