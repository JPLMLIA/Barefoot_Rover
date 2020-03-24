//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: random_forest.h
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Structures and prototypes
// ========================================================================
#pragma once
#include <stdbool.h>

#include "decision_tree.h"

//#define INVERT_TEST_DATA // If enabled, use [feature][sample] instead of [sample][feature] for test data
// Nodes will also be co-allocated based on which feature they test

// Hyper parameters
replicated uint64_t g_numTrees;	// tree count in file
replicated uint64_t g_treeMax;	// max number of trees to actually use
replicated uint64_t g_sampleMax; // max number of test samples to actually use
replicated uint64_t g_treesPerThread;	// # trees handled by a sample thread
replicated uint64_t g_selectionSize; // Number of random features picked at each split
//replicated uint64_t g_trainPercent; // Rest is test
// end of hyper parameters

replicated uint64_t g_numTrainSamples;
replicated uint64_t g_numTestSamples;
replicated uint64_t g_numFeatures;
replicated uint64_t g_numClasses;

replicated double **g_trainSamples; // [numTrainSamples][numFeatures]
replicated uint64_t *g_trainSamplesLabels; // [numTrainSamples]

replicated double **g_testSamples; // [numFeatures][numTestSamples]
replicated uint64_t *g_testSamplesLabels; // [numTestSamples]

replicated uint64_t **g_testSamplesLabelsVotes; // [numTestSamples][numClasses]
replicated uint64_t **g_testSamplesClassesProbs; // [numClasses]

replicated Node **g_trees; // [numTrees]

void initTrees();

void train();
void trainTree(uint64_t);

void test();

struct candidate_node_t;
typedef struct candidate_node_t CandidateNode;
Node *allocateNodeFromCandidate(Node *parent, CandidateNode *, bool branch);
Node *makeLeaf(Node *parent, CandidateNode *, bool branch);
Node *allocateLeafFromCandidate(Node *parent, CandidateNode *, bool branch);

Node* allocateNode(void *homePtr);

// Test all the samples: s[featureIndex] < threshold
void performTest(Node *);

