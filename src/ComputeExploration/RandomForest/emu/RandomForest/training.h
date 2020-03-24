//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: training.h
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Structures and prototypes used during the training of Random Forest
// ========================================================================
#pragma once

// Should we launch one thread per threshold?
// If many trees, leave commented, no benefit added from doing thresholds in parallel
//#define PARALLELIZE_GINI_THRESHOLDS

//#define VECTOR_OUTPUTS

//#define USE_BAGGING

//#include <unordered_set>
#include <set>
#ifdef VECTOR_OUTPUTS
#include <vector>
#endif

#include "rfc_api.h"

typedef struct training_state_t {
	MaxFeatures m_maxFeatures;
} TrainingState;

TrainingState g_trainingState;

// Hyper parameters
// end of hyper parameters

// Should this be declared for each yes/no for each possible feature/threshold being tested?
// In this case the only things "shown here" would be those that were discriminated as "true" if this is the trueBranch (resp false) of the parent
// Once a decision as been made based on the Gini index, we free the unused ones

// Samples references in a node are stored as indices, ordered by the current feature for a given node
typedef struct sampleIndex_t SampleIndex;
struct sampleIndex_t {
	uint64_t      m_index;
//	double       *m_sample;
	double        m_value;
	//SampleIndex  *m_previous;
	uint64_t      m_weightedVote;
	SampleIndex  *m_next;
};

typedef struct candidate_node_t CandidateNode; 

struct candidate_node_t {
	uint64_t  m_featureIndex;
	double    m_threshold; // computed by checking which of the average between consequent (ordered) features has the lowest Gini impurity
	double    m_giniImpurity;

	std::set<uint64_t> *m_availableFeatures; // set is faster than unordered_set

	// Input: linked list, ordered based on the value
	SampleIndex *m_inputSamples;
	uint64_t m_inputSamplesCount;

	// True branch
#ifndef VECTOR_OUTPUTS
//#	ifndef USE_BAGGING
	uint64_t  m_trueSamplesCount;
//#	else
#	ifdef USE_BAGGING
	uint64_t  m_trueSamplesCumulatedWeightedVotes;
	//uint64_t *m_trueSamplesWeights;
#	endif
	uint64_t *m_trueSamples; // Initialized to input size
#else
	std::vector<uint64_t> *m_trueSamplesVector;
#endif
#ifndef USE_BAGGING
	uint64_t *m_trueClassesCount; // [numclasses]
#else
	uint64_t *m_trueClassesWeightedVotes; // [numclasses]
#endif

	// False branch
#ifndef VECTOR_OUTPUTS
//#	ifndef USE_BAGGING
	uint64_t  m_falseSamplesCount;
//#	else
#	ifdef USE_BAGGING
	uint64_t  m_falseSamplesCumulatedWeightedVotes;
	//uint64_t *m_falseSamplesWeights;
#	endif
	uint64_t *m_falseSamples; // Initialized to input size
#else
	std::vector<uint64_t> *m_falseSamplesVector;
#endif
#ifndef USE_BAGGING
	uint64_t *m_falseClassesCount; // [numclasses]
#else
	uint64_t *m_falseClassesWeightedVotes; // [numclasses]
#endif
};

replicated uint64_t **g_samplesWeights; // [numTrees][numTrainingSamples]

void parseFile_sklearn(const char *filename);

// Atomic ordered insertion!
void insertSample(CandidateNode *, uint64_t sampleIndex, double sampleValue);

void initTrees();

void train();
void train_spawnRecursive(uint64_t, uint64_t, void *);
void trainTree(uint64_t);
void readTree(uint64_t t, int nodeNum, FILE* modelFile);
void readTreeMigrate(uint64_t t, int nodeNum, FILE* modelFile);

struct node_t;
typedef struct node_t Node;


CandidateNode *allocateCandidateNode(void *homePtr, uint64_t featureIndex, uint64_t samplesCount, CandidateNode *parent);
void freeCandidateNode(CandidateNode *);

CandidateNode *createChildNode(uint64_t tree, CandidateNode *parent, uint64_t indexFeature, bool branch);

// Test all the samples: s[featureIndex] < threshold
void performTest(Node *);

// This computes the Gini index, based on the pre-computed threshold:
// It fills the samples lists for each true and false branches
// It counts the number of these samples for each class based on their labels
void computeGini(CandidateNode *);

// This uses the fact that we keep the samples ordered!
void computeThreshold(CandidateNode **);
//double computeGiniThreshold(CandidateNode *, double, CandidateNode *);
double computeGiniThreshold(CandidateNode *, double, uint64_t, CandidateNode *);
void computeGiniThreshold_thread(void *unused, CandidateNode **node, double avg, uint64_t count, CandidateNode **chosen, double *lowestGini);

#ifndef USE_BAGGING
void trainBranch_recursive(Node *parent, CandidateNode *cparent, bool branch);
void insertInput(CandidateNode *node, uint64_t sampleIndex, double value);
void insertOutput(CandidateNode *node, bool branch, uint64_t sampleIndex, uint64_t label);
void fillInput(CandidateNode *parent, CandidateNode *child, bool branch);
#else
void trainBranch_recursive(Node *parent, CandidateNode *cparent, uint64_t tree, bool branch);
void insertInput(CandidateNode *node, uint64_t sampleIndex, double value, uint64_t weight);
void insertOutput(CandidateNode *node, bool branch, uint64_t sampleIndex, uint64_t label, uint64_t weight);
//void insertOutput(CandidateNode *node, bool branch, uint64_t sampleIndex, uint64_t label);
void fillInput(CandidateNode *parent, CandidateNode *child, uint64_t tree, bool branch);
#endif

// DEBUG functions
void printInput(CandidateNode *);

