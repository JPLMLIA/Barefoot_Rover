//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: random_forest_classifier.cpp
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Random Forest first implementation
// ========================================================================

#include <float.h>
#include <limits>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sstream>
#include <vector>

extern "C" {
#include <cilk.h>
#include <memoryweb.h>
}

#include "rfc_api.h"
#include "random_forest_classifier.h"
#include "log.h"
#include "training.h"

/*
int main(int argc, char **argv) {
	char filename[4096];
	if (argc < 5) {
		fprintf(stderr, "Argument needed: <data file name> <num_trees> <selection_size> <training ratio>\n");
		return -1;
	}
	strcpy(filename, argv[1]);
	//mw_replicated_init((long *)&g_numTrees, atol(argv[2]));
	//mw_replicated_init((long *)&g_selectionSize, atol(argv[3]));

	randomForestClassifier_initialize(atol(argv[2]), AUTO, 0);

	mw_replicated_init((long *)&g_trainPercent, 100 * atof(argv[4]));

	parseFile_sklearn(filename);

	starttiming();
	volatile uint64_t starttimer = CLOCK();

	initTrees();

	//train();
	randomForestClassifier_train();

	g_rightResults = 0;
	g_wrongResults = 0;
	g_rightResults = 1;

	test();

	volatile uint64_t endtimer = CLOCK();
	uint64_t delta = endtimer - starttimer;
	LOG_INFO("Completed in %lu cycles Right %lu vs Wrong %lu (accuracy %.2f%%)\n", delta, g_rightResults, g_wrongResults, (double)g_rightResults / (double)(g_rightResults + g_wrongResults) * 100.0);

	return 0;
}

// Assuming the following format:
// First line: number of entries,number of features per line
// All others: CSV features. We assume only the useful ones have already been selected
void parseFile_sklearn(const char *filename) {
	FILE *fd = fopen(filename, "r");
	if (!fd) {
		fprintf(stderr, "Could not open data file %s\n", filename);
		exit(-1);
	}
	uint64_t nsamp, nf;
	fscanf(fd, "%lu,%lu,", &nsamp, &nf);
	uint64_t ntrain = nsamp * ((double)g_trainPercent / 100.0);
	uint64_t ntest = nsamp - ntrain;
	//mw_replicated_init((long *)&g_numTrainSamples, (long)ntrain);
	mw_replicated_init((long *)&g_numTestSamples, (long)ntest);
	//mw_replicated_init((long *)&g_numFeatures, (long)nf);
	uint64_t nc = 1;
	while (true) {
		char c;
		fread(&c, 1, sizeof(char), fd);
		if (c == ',')
			nc++;
		if (c == '\n')
			break;
	}
//	mw_replicated_init((long *)&g_numClasses, (long)nc);
	randomForestClassifier_initializeTraining(ntrain, nf, nc);

	LOG_INFO("=== Reading %s #samples: %lu %lu #features: %lu #classes: %lu Number trees= %lu Selection size= %lu ===\n", filename, g_numTrainSamples, g_numTestSamples, g_numFeatures, g_numClasses, g_numTrees, g_selectionSize);


#ifdef INVERT_TEST_DATA
	double **samplesTestTmp = (double **)mw_malloc2d(nf, ntest * sizeof(double));
#else
	double **samplesTestTmp = (double **)mw_malloc2d(ntest, nf * sizeof(double));
#endif
	mw_replicated_init((long *)&g_testSamples, (long)samplesTestTmp);

	double **samplesWeightsTmp = (double **)mw_malloc2d(g_numTrees, ntrain * sizeof(double));
	mw_replicated_init((long *)&g_samplesWeights, (long)samplesWeightsTmp);

	uint64_t *samplesLabelsTestTmp = (uint64_t *)mw_malloc1dlong(ntest);
	mw_replicated_init((long *)&g_testSamplesLabels, (long)samplesLabelsTestTmp);

	uint64_t **samplesTestClassesTmp = (uint64_t **)mw_malloc2d(ntest, g_numClasses * sizeof(uint64_t));
	mw_replicated_init((long *)&g_testSamplesClassesVotes, (long)samplesTestClassesTmp);

	for (uint64_t s = 0; s < ntest; s++)
		memset(g_testSamplesClassesVotes, 0 , g_numClasses * sizeof(uint64_t));

	double *gc_samples = (double *)malloc(ntrain * nf * sizeof(double));
	uint64_t *gc_classes = (uint64_t *)malloc(ntrain * sizeof(uint64_t));

	uint64_t foldValue = nsamp / g_numTestSamples;
	LOG_DEBUG("folded: %lu\n", foldValue);
	uint64_t ntr = 0; // training
	uint64_t ntt = 0; // test
	for (uint64_t s = 0; s < nsamp; s++) {
		bool testing = (s % foldValue == 0);
		//if (testing) {
		//	LOG_DEBUG("Testing: %lu %lu\n", s, ntt);
		//}
		for (uint64_t f = 0; f < g_numFeatures; f++) {
			double val;
			fscanf(fd, "%lf,", &val);
			if (!testing) {
				gc_samples[ntr * nf + f] = val;
	//			g_trainSamples[ntr][f] = val;
			} else
#ifdef INVERT_TEST_DATA
				g_testSamples[f][ntt] = val;
#else
				g_testSamples[ntt][f] = val;
#endif
		}
		uint64_t label;
		fscanf(fd, "%lu\n", &label);
		if (!testing) {
			gc_classes[ntr] = label;
	//		g_trainSamplesLabels[ntr] = label;
			ntr++;
		} else
			g_testSamplesLabels[ntt++] = label;
	}
	fclose(fd);

	randomForestClassifier_fillTrainingData(gc_samples, gc_classes);
	free(gc_samples);
	free(gc_classes);
}

*/
void initTrees() {
	Node *treesTmp = (Node *)mw_malloc1dlong(g_numTrees);
	mw_replicated_init((long *)&g_trees, (long)treesTmp);
	LOG_DEBUG("tree 0,1 at @%p, @%p\n", g_trees[0], g_trees[1]);
	LOG_DEBUG("ptree 0,1 at @%p, @%p\n", &g_trees[0], &g_trees[1]);
}

void train() {
	//for (uint64_t t = 0; t < g_numTrees; t++) {
/*	
	for (uint64_t t = 0; t < 1; t++) {
		LOG_DEBUG("Training tree #%lu\n", t);
		trainTree(t);
	}
*/	
	train_spawnRecursive(0, NODELETS(), g_trainSamples[0]);
}

void train_spawnRecursive(uint64_t low, uint64_t high, void *dummy) {
    while(true) {
        uint64_t count = high - low;
        if (count <= 1)
            break;
        uint64_t mid = low + count / 2;
        cilk_spawn train_spawnRecursive(mid, high, g_trainSamples[mid]);
        high = mid;
    }   

    for (uint64_t k = low; k < g_numTrees; k += NODELETS()) 
        cilk_spawn trainTree(k);
			    
    cilk_sync;
}

void trainTree(uint64_t t) {
	// Step 1: pick the root node randomly
	int rootIndex = (double)rand() / (double)RAND_MAX * g_numFeatures;
	LOG_DEBUG("Tree #%lu has root node with feature %lu\n", t, rootIndex);

#ifndef USE_BAGGING
	CandidateNode *rootCNode = allocateCandidateNode(g_trees[t], rootIndex, g_numTrainSamples, NULL);
	for (uint64_t s = 0; s < g_numTrainSamples; s++) {
		//LOG_DEBUG("Inserting: %lu\n", s);
		double val = g_trainSamples[s][rootIndex];
		insertInput(rootCNode, s, val);
		//LOG_DEBUG("Inserted: %lu\n", s);
	}
#else
	// Generate bootstrapping
	std::vector<uint64_t> indices;
	uint64_t *indicesWeights = new uint64_t[g_numTrainSamples];
	memset(indicesWeights, 0, g_numTrainSamples * sizeof(uint64_t));
	for (uint64_t i = 0; i < g_numTrainSamples; i++) {
		uint64_t idx = (double)rand() / (double)RAND_MAX * g_numTrainSamples;
		//LOG_DEBUG("t%02lu picked: %d\n", t, idx);
		indicesWeights[idx]++;
	}
	uint64_t numTrainSamples = 0; // Aka nnz inside indicesWeights
	for (uint64_t i = 0; i < g_numTrainSamples; i++) {
		if (indicesWeights[i] > 0)
			numTrainSamples++;
	}
	LOG_INFO("Tree #%lu has %lu/%lu samples in its bag for training\n", t, numTrainSamples, g_numTrainSamples);
	CandidateNode *rootCNode = allocateCandidateNode(&g_trees[t], rootIndex, numTrainSamples, NULL);
	for (uint64_t s = 0; s < g_numTrainSamples; s++) {
		g_samplesWeights[t][s] = indicesWeights[s];
		if (indicesWeights[s] == 0)
			continue;
		double val = g_trainSamples[s][rootIndex];
		insertInput(rootCNode, s, val, indicesWeights[s]);
	}
	delete [] indicesWeights;
#endif

	//printInput(rootCNode);

	// Step 2: choose the threshold: this gets us the lowest Gini!
	rootCNode->m_availableFeatures = new std::set<uint64_t>();
	for (uint64_t f = 0; f < g_numFeatures; f++) {
		if (f == rootIndex)
			continue;
		rootCNode->m_availableFeatures->insert(f);
	}
	computeThreshold(&rootCNode);
	//rootCNode->m_featuresIndicesHistory[rootCNode->m_featureIndex] = 1;
	//rootCNode->m_featuresDepth++;

#ifndef VECTOR_OUTPUTS
#	ifndef USE_BAGGING
	LOG_INFO("*** Root node picking feature index %lu, threshold %f (gini %f) split: [%lu|%lu] ***\n", rootCNode->m_featureIndex, rootCNode->m_threshold, rootCNode->m_giniImpurity, rootCNode->m_trueSamplesCount, rootCNode->m_falseSamplesCount);
#	else
	LOG_INFO("*** Root node picking feature index %lu, threshold %f (gini %f) split: [%lu|%lu] ***\n", rootCNode->m_featureIndex, rootCNode->m_threshold, rootCNode->m_giniImpurity, rootCNode->m_trueSamplesCumulatedWeightedVotes, rootCNode->m_falseSamplesCumulatedWeightedVotes);
#	endif
#else
	LOG_INFO("*** Root node picking feature index %lu, threshold %f (gini %f) split: [%lu|%lu] ***\n", rootCNode->m_featureIndex, rootCNode->m_threshold, rootCNode->m_giniImpurity, rootCNode->m_trueSamplesVector->size(), rootCNode->m_falseSamplesVector->size());
#endif

	// Step 3: split between true and false nodes!
	Node *rootNode = allocateNodeFromCandidate(NULL, rootCNode, true);
	g_trees[t] = rootNode;
	//LOG_DEBUG("t? %lu set to @%p\n", t, rootNode);
#ifndef USE_BAGGING
	cilk_spawn trainBranch_recursive(rootNode, rootCNode, true);
	cilk_spawn trainBranch_recursive(rootNode, rootCNode, false);
#else
	cilk_spawn trainBranch_recursive(rootNode, rootCNode, t, true);
	cilk_spawn trainBranch_recursive(rootNode, rootCNode, t, false);
#endif
	cilk_sync;

	freeCandidateNode(rootCNode);
}
void readTree(uint64_t t, int nodeNum, FILE* modelFile) {
	readTreeMigrate(t, nodeNum, modelFile);
	//cilk_spawn readTreeMigrate(t, nodeNum, modelFile);
	// implicit cilk_sync at end of function
}

//noinline
void readTreeMigrate(uint64_t t, int nodeNum, FILE* modelFile) {
	float	clas;
	int		featureIndex = 0;
	int		isLeaf = -1;
	Node	**nodeTable;
	Node	*node;
	float	prob;
	float	threshold = 0.0;

	// nodeTable is a temporary construct for linking parents with children
	// and detecting empty spots in the tree.
	// malloc forces migration to this tree's nodelet
	//	LOG_DEBUG("Starting for tree %d out of %d\n", t, g_numTrees);
	nodeTable = (Node **)mw_localmalloc(sizeof(Node*) * nodeNum, &g_trees[t]);
	memset(nodeTable,NULL,sizeof(Node*) * nodeNum);

	for (int j=0; j < nodeNum; j++)
		{
		// we are still in sequential mode, for file reading
		if ((nodeTable[j] == NULL) && (j > 0))
			{
			continue;
			}
		node = allocateNode(&g_trees[t]);
		fread(&isLeaf,sizeof(int),1,modelFile);

		if (j > 0)
			// set up link for this node's parents
			{
			if ((j & 1) == 1)
				// odd numbers are left (true) child
				nodeTable[j] -> m_trueBranch = node;
			else
				nodeTable[j] -> m_falseBranch = node;
			}
	
		// Read of isLeaf forces migration back to node 0.
		// isLeaf is forced to node 0 because of the i/o call fread()
		// To avoid migration generally, use "noinline" keyword.
		// See programming manual 6.3.
		// BUT DOES NOT WORK IN THIS CASE
		if (isLeaf == 0)
			{
			if ((j*2+1) > 1022)
				LOG_DEBUG("### j too big\n");
			nodeTable[j*2+1] = node;	// save node as parent for child node
			nodeTable[j*2+2] = node;
			fread(&featureIndex,sizeof(int),1,modelFile);
			fread(&threshold,sizeof(float),1,modelFile);
			node->_isLeaf = 0;
			node->m_featureIndex = featureIndex;
			node->m_threshold = threshold;
			}
		else
			{
			fread(&clas,sizeof(float),1,modelFile);
			fread(&prob,sizeof(float),1,modelFile);
			node->_isLeaf = 1;
			node->_class = clas;
			node->_prob = prob;
			}
		if (j == 0)
			{	// root node of tree t
			g_trees[t] = node;
			if (t == 0)
				LOG_DEBUG("tree 0 now at @%p\n", g_trees[0]);
			}
		}	// for j...
	mw_localfree(nodeTable);
	//free(nodeTable);
}

#ifndef USE_BAGGING
void trainBranch_recursive(Node *parent, CandidateNode *cparent, bool branch) {
#else
void trainBranch_recursive(Node *parent, CandidateNode *cparent, uint64_t tree, bool branch) {
#endif
	//LOG_DEBUG("Train depth %lu recursive branch %s\n", (g_numFeatures - parent->m_availableFeatures->size()), (branch ? "true" : "false"));
	double lowestGini = 1.0;
	CandidateNode *cnode = NULL;
	// Iterate over the remaining features
	// "cilk_for"
	if (cparent->m_availableFeatures->size() == 0) {
		LOG_CRITICAL("Cannot find available features to choose from!\n", 1);
		exit(-1);
	}
	std::set<uint64_t> selection = *(cparent->m_availableFeatures);
	for (uint64_t i = 0; i < std::min(g_selectionSize,  cparent->m_availableFeatures->size()); i++) {
		uint64_t j = (double)rand() / (double)RAND_MAX * selection.size();
		std::set<uint64_t>::iterator it = selection.begin();
		for (uint64_t k = 0; k < j; ++it, k++) {} // Faster than std::advance
		//std::advance(it, j);
		uint64_t f = *it;
		selection.erase(it);
#ifndef VECTOR_OUTPUTS
//#	ifndef USE_BAGGING
		CandidateNode *node = allocateCandidateNode(cparent, f, (branch ? cparent->m_trueSamplesCount : cparent->m_falseSamplesCount), cparent);
//#	else
//		CandidateNode *node = allocateCandidateNode(cparent, f, (branch ? cparent->m_trueSamplesCumulatedWeightedVotes : cparent->m_falseSamplesCumulatedWeightedVotes), cparent);
//#	endif
#else
		CandidateNode *node = allocateCandidateNode(cparent, f, (branch ? cparent->m_trueSamplesVector : cparent->m_falseSamplesVector)->size(), cparent);
#endif
#ifndef USE_BAGGING
		fillInput(cparent, node, branch);
#else
		fillInput(cparent, node, tree, branch);
#endif
		node->m_availableFeatures = new std::set<uint64_t>();
		for (std::set<uint64_t>::iterator it = cparent->m_availableFeatures->begin(); it != cparent->m_availableFeatures->end(); ++it) {
			if (*it == node->m_featureIndex)
				continue;
			node->m_availableFeatures->insert(*it);
		}
		computeThreshold(&node);

		if (node->m_giniImpurity < lowestGini) {
			lowestGini = node->m_giniImpurity;
			if (cnode != NULL)
				freeCandidateNode(cnode);
			cnode = node;
		}
	}
	if (cnode == NULL) {
		LOG_CRITICAL("Leaf found (no features!)\n", 1);
		makeLeaf(parent, cparent, branch);
		return;
	}
	// *cnode->m_availableFeatures = *parent->m_availableFeatures;
	//cnode->m_availableFeatures->erase(cnode->m_featureIndex);

/*
	char lf[4096];
	sprintf(lf, "[");
	for (uint64_t i = 0; i < g_numFeatures; i++) {
		if (cnode->m_featuresIndicesHistory[i]) {
			char tmp[8];
			sprintf(tmp, "%lu ", i);
			strcat(lf, tmp);
		}
	}
	//sprintf(lf, "%s]", lf);
	strcat(lf, "]");
	*/
	/*
	std::stringstream ss;
	ss << "[ ";
	for (uint64_t f = 0; f < g_numFeatures; f++) {
	//for (std::set<uint64_t>::iterator it = cnode->m_availableFeatures.begin(); it != cnode->m_availableFeatures.end(); it++) {
		if (cnode->m_availableFeatures->find(f) == cnode->m_availableFeatures->end())
			ss << f << " ";
	}
	ss<< "]";
	*/
#ifndef VECTOR_OUTPUTS
#	ifndef USE_BAGGING
	//LOG_INFO("*** Depth %lu Branch %s, picking feature index %lu, threshold %f (gini %f) split: [%lu|%lu] Features: %s ***\n", (g_numFeatures - cnode->m_availableFeatures->size()), (branch ? "true" : "false"), cnode->m_featureIndex, cnode->m_threshold, cnode->m_giniImpurity, cnode->m_trueSamplesCount, cnode->m_falseSamplesCount, ss.str().c_str());
#	else
	//LOG_INFO("*** Depth %lu Branch %s, picking feature index %lu, threshold %f (gini %f) split: [%lu|%lu] Features: %s ***\n", (g_numFeatures - cnode->m_availableFeatures->size()), (branch ? "true" : "false"), cnode->m_featureIndex, cnode->m_threshold, cnode->m_giniImpurity, cnode->m_trueSamplesCumulatedWeightedVotes, cnode->m_falseSamplesCumulatedWeightedVotes, ss.str().c_str());
#	endif
#else
	//LOG_INFO("*** Depth %lu Branch %s, picking feature index %lu, threshold %f (gini %f) split: [%lu|%lu] Features: %s ***\n", (g_numFeatures - cnode->m_availableFeatures->size()), (branch ? "true" : "false"), cnode->m_featureIndex, cnode->m_threshold, cnode->m_giniImpurity, cnode->m_trueSamplesVector->size(), cnode->m_falseSamplesVector->size(), ss.str().c_str());
#endif

	if (cnode->m_giniImpurity >= cparent->m_giniImpurity) {
		//LOG_INFO("We have a leaf (Gini) for branch %s\n", (branch ? "true" : "false"));
		//allocateLeafFromCandidate(node, cnode, branch);
		allocateLeafFromCandidate(parent, cnode, branch);
		freeCandidateNode(cnode);
		return;
	}
	//if (cnode->m_featuresDepth == g_numFeatures) {
	if (cnode->m_availableFeatures->size() == 0) {
		//LOG_INFO("We have a leaf (no more features) for branch %s\n", (branch ? "true" : "false"));
		//allocateLeafFromCandidate(node, cnode, branch);
		allocateLeafFromCandidate(parent, cnode, branch);
		freeCandidateNode(cnode);
		return;
	}

	Node *node = allocateNodeFromCandidate(parent, cnode, branch);

#ifndef VECTOR_OUTPUTS
	if (cnode->m_trueSamplesCount > 1) {
#	ifndef USE_BAGGING
		cilk_spawn trainBranch_recursive(node, cnode, true);
#	else 
		cilk_spawn trainBranch_recursive(node, cnode, tree, true);
#	endif
	} else {
		//LOG_INFO("We have a leaf (%lu samples) on branch true\n", cnode->m_trueSamplesCount);
		allocateLeafFromCandidate(parent, cnode, branch);
		//allocateLeafFromCandidate(node, cnode, branch);
	}
#else
	if (cnode->m_trueSamplesVector->size() > 1) {
		cilk_spawn trainBranch_recursive(node, cnode, true);
	} else 
		//LOG_INFO("We have a leaf (%lu samples) on branch true\n", cnode->m_trueSamplesVector->size());
#endif

#ifndef VECTOR_OUTPUTS
	if (cnode->m_falseSamplesCount > 1) {
#	ifndef USE_BAGGING
		cilk_spawn trainBranch_recursive(node, cnode, false);
#	else
		cilk_spawn trainBranch_recursive(node, cnode, tree, false);
#	endif
	} else {
		//LOG_INFO("We have a leaf (%lu samples) on branch false\n", cnode->m_falseSamplesCount);
		//allocateLeafFromCandidate(node, cnode, branch);
		allocateLeafFromCandidate(parent, cnode, branch);
	}
#else
	if (cnode->m_falseSamplesVector->size() > 1) {
		cilk_spawn trainBranch_recursive(node, cnode, false);
	} else
		//LOG_INFO("We have a leaf (%lu samples) on branch false\n", cnode->m_falseSamplesVector->size());
#endif
	cilk_sync;

	freeCandidateNode(cnode);
}

Node *allocateNodeFromCandidate(Node *parent, CandidateNode *cnode, bool branch) {
#ifdef INVERT_TEST_DATA
	Node *node = (Node *)mw_localmalloc(sizeof(Node), g_testSamples[cnode->m_featureIndex]);
#else
	Node *node = (Node *)mw_localmalloc(sizeof(Node), cnode);
#endif
	node->m_featureIndex = cnode->m_featureIndex;
	node->m_threshold = cnode->m_threshold;
	node->m_trueBranch = NULL;
	node->m_falseBranch = NULL;
	if (parent != NULL) {
		if (branch)
			parent->m_trueBranch = node;
		else
			parent->m_falseBranch = node;
	} 
	node->m_labelsVotes = NULL;
	//LOG_DEBUG("new node: @%p parent was: @%p\n", node, parent);
	//if (parent)
	//	LOG_DEBUG("Parents children: @%p @%p\n", parent->m_trueBranch, parent->m_falseBranch);
	return node;
}

Node *allocateLeafFromCandidate(Node *parent, CandidateNode *cnode, bool branch) {
	//LOG_DEBUG("allocate leaf from candidate\n", cnode->m_featureIndex);
#ifdef INVERT_TEST_DATA
	Node *node = (Node *)mw_localmalloc(sizeof(Node), g_testSamples[cnode->m_featureIndex]);
#else
	Node *node = (Node *)mw_localmalloc(sizeof(Node), cnode);
#endif
	node->m_threshold = std::numeric_limits<double>::quiet_NaN();
	node->m_trueBranch = NULL;
	node->m_falseBranch = NULL;
	if (parent != NULL) {
		if (branch)
			parent->m_trueBranch = node;
		else
			parent->m_falseBranch = node;
	} 
	node->m_labelsVotes = (uint64_t *)mw_localmalloc(g_numClasses * sizeof(uint64_t), (long *)node);
	//std::stringstream labels;
	//labels << "[";
	for (uint64_t i = 0; i < g_numClasses; i++) {
#ifndef USE_BAGGING
		uint64_t classesVotes = cnode->m_trueClassesCount[i] + cnode->m_falseClassesCount[i];
#else
		uint64_t classesVotes = cnode->m_trueClassesWeightedVotes[i] + cnode->m_falseClassesWeightedVotes[i];
#endif
		node->m_labelsVotes[i] = classesVotes;
	//	labels << " " << classesVotes;
	}
	//labels << " ]";
	//LOG_DEBUG("Created new leaf @%p-> @%p. Labels votes is: %s\n", node, node->m_labelsVotes, labels.str().c_str());
	return node;
}

Node *makeLeaf(Node *node, CandidateNode *cnode, bool branch) {
	 //cnode->m_inputSamples->m_index;
	//node->m_featureIndex = cnode->m_inputSamples->m_index;
	LOG_DEBUG("TODO: make leaf!\n");
	return node;
}

CandidateNode *allocateCandidateNode(void *homePtr, uint64_t index, uint64_t samplesCount, CandidateNode *parent) {
	CandidateNode *node = (CandidateNode *)mw_localmalloc(sizeof(CandidateNode), homePtr);
	node->m_featureIndex = index;
	node->m_inputSamplesCount = 0;
	node->m_inputSamples = NULL;
#ifndef VECTOR_OUTPUTS
	node->m_trueSamplesCount = 0;
#	ifdef USE_BAGGING
	node->m_trueSamplesCumulatedWeightedVotes = 0;
#	endif
	node->m_trueSamples = (uint64_t *)mw_localmalloc(samplesCount * sizeof(uint64_t), node);
#else
	node->m_trueSamplesVector = new std::vector<uint64_t>();
#endif

#ifndef USE_BAGGING
	node->m_trueClassesCount = (uint64_t *)mw_localmalloc(g_numClasses * sizeof(uint64_t), node);
	memset(node->m_trueClassesCount, 0, g_numClasses * sizeof(uint64_t));
#else
	node->m_trueClassesWeightedVotes = (uint64_t *)mw_localmalloc(g_numClasses * sizeof(uint64_t), node);
	memset(node->m_trueClassesWeightedVotes, 0, g_numClasses * sizeof(uint64_t));
#endif
#ifndef VECTOR_OUTPUTS
	node->m_falseSamplesCount = 0;
#	ifdef USE_BAGGING
	node->m_falseSamplesCumulatedWeightedVotes = 0;
#	endif
	node->m_falseSamples = (uint64_t *)mw_localmalloc(samplesCount * sizeof(uint64_t), node);
#else
	node->m_falseSamplesVector = new std::vector<uint64_t>();
#endif

#ifndef USE_BAGGING
	node->m_falseClassesCount = (uint64_t *)mw_localmalloc(g_numClasses * sizeof(uint64_t), node);
	memset(node->m_falseClassesCount, 0, g_numClasses * sizeof(uint64_t));
#else
	node->m_falseClassesWeightedVotes = (uint64_t *)mw_localmalloc(g_numClasses * sizeof(uint64_t), node);
	memset(node->m_falseClassesWeightedVotes, 0, g_numClasses * sizeof(uint64_t));
#endif

	//node->m_featuresIndicesHistory = (uint64_t *)mw_localmalloc(g_numFeatures * sizeof(uint64_t), node);
	//LOG_DEBUG("Allocate memory @%p\n", node);
	//node->m_availableFeatures = new std::set<uint64_t>();
	node->m_availableFeatures = NULL;
	//LOG_DEBUG("After set initialization?? @%p\n", node->m_availableFeatures);
	if (parent != NULL)  {
		//LOG_DEBUG("Copying %lu feature indices in history\n",  parent->m_featuresDepth);
		//memcpy(node->m_featuresIndicesHistory, parent->m_featuresIndicesHistory, g_numFeatures * sizeof(uint64_t));
		//node->m_featuresDepth = parent->m_featuresDepth;
		//node->m_availableFeatures = parent->m_availableFeatures;
	} else {
		//memset(node->m_featuresIndicesHistory, 0, g_numFeatures * sizeof(uint64_t));
		//node->m_featuresDepth = 0;
	}
	//LOG_DEBUG("Allocate memory\n", 1);
	//node->m_availableFeatures.erase(node->m_featureIndex);
	//LOG_DEBUG("allocated @%p, with sample counts: %lu\n", node, samplesCount);
	return node;
}

Node* allocateNode(void *homePtr) {

	Node *node = (Node *)mw_localmalloc(sizeof(Node), homePtr);
	return node;

}

void freeCandidateNode(CandidateNode *node) {
#ifndef VECTOR_OUTPUTS
	if (node->m_trueSamples != NULL)
		mw_localfree(node->m_trueSamples);
#else
	delete node->m_trueSamplesVector;
#endif
#ifndef USE_BAGGING
	mw_localfree(node->m_trueClassesCount);
#else
	mw_localfree(node->m_trueClassesWeightedVotes);
#endif
#ifndef VECTOR_OUTPUTS
	if (node->m_falseSamples != NULL)
		mw_localfree(node->m_falseSamples);
#else
	delete node->m_falseSamplesVector;
#endif
#ifndef USE_BAGGING
	mw_localfree(node->m_falseClassesCount);
#else
	mw_localfree(node->m_falseClassesWeightedVotes);
#endif
	SampleIndex *header = node->m_inputSamples;
	while (header != NULL) {
		SampleIndex *ptr = header;
		header = header->m_next;
		mw_localfree(ptr);
	}
	//mw_localfree(node->m_featuresIndicesHistory);
	if (node->m_availableFeatures != NULL)
		delete node->m_availableFeatures;
	mw_localfree(node);
}

void performTest(Node *node) {
	/*
	// Save this in parameters to make sure they migrate lightly!
	uint64_t f = node->m_featureIndex;
	double   threshold = node->m_threshold;
	for (uint64_t i = 0; i < node->m_samplesCount; i++) {
		uint64_t s = node->m_samplesIndices[i];
		//LOG_DEBUG("s=%lu idx: %lu val=%f label:%lu\n", s, f, g_samples[s][f], g_samplesLabels[s]);
		// This will migrate to the nodelet containing the sample
		if (g_samples[s][f] < threshold) {
			ATOMIC_ADDM((long *)&(node->m_trueBranch->m_samplesClasses[g_samplesLabels[s]]), 1);
		} else {
			ATOMIC_ADDM((long *)&(node->m_falseBranch->m_samplesClasses[g_samplesLabels[s]]), 1);
		}
	}

	for (uint64_t i = 0; i < g_numClasses; i++) {
		LOG_DEBUG("Arity: %lu true: %lu false: %lu\n", i, node->m_trueBranch->m_samplesClasses[i], node->m_falseBranch->m_samplesClasses[i]);
	}
	*/
}

void computeGini(CandidateNode *node) {
	double trueBranchGini = 1.0;
	//LOG_DEBUG("@%p true count: %lu false count: %lu [%lu %lu %lu] [%lu %lu %lu]\n", node, node->m_trueSamplesVector->size(), node->m_falseSamplesVector->size(), 
	//		node->m_trueClassesCount[0], node->m_trueClassesCount[1],node->m_trueClassesCount[2],
	//		node->m_falseClassesCount[0], node->m_falseClassesCount[1], node->m_falseClassesCount[2]);
#ifndef VECTOR_OUTPUTS
#	ifndef USE_BAGGING
	if (node->m_trueSamplesCount != 0) {
#	else
	if (node->m_trueSamplesCumulatedWeightedVotes != 0) {
#	endif
		for (uint64_t i = 0; i < g_numClasses; i++) {
#	ifndef USE_BAGGING
			double val = (double)node->m_trueClassesCount[i] / (double)node->m_trueSamplesCount;
#	else
			double val = (double)node->m_trueClassesWeightedVotes[i] / (double)node->m_trueSamplesCumulatedWeightedVotes;
#	endif
			trueBranchGini -= val * val;
		}
	}
#else
	if (node->m_trueSamplesVector->size() != 0) {
		for (uint64_t i = 0; i < g_numClasses; i++) {
			double val = (double)node->m_trueClassesCount[i] / (double)node->m_trueSamplesVector->size();
			trueBranchGini -= val * val;
		}
	} 
#endif

	double falseBranchGini = 1.0;
#ifndef VECTOR_OUTPUTS
#	ifndef USE_BAGGING
	if (node->m_falseSamplesCount != 0) {
#	else
	if (node->m_falseSamplesCumulatedWeightedVotes != 0) {
#	endif
		for (uint64_t i = 0; i < g_numClasses; i++) {
#	ifndef USE_BAGGING
			double val = (double)node->m_falseClassesCount[i] / (double)node->m_falseSamplesCount;
#	else
			double val = (double)node->m_falseClassesWeightedVotes[i] / (double)node->m_falseSamplesCumulatedWeightedVotes;
#	endif
			falseBranchGini -= val * val;
		}
	}
#	ifndef USE_BAGGING
	node->m_giniImpurity = (node->m_trueSamplesCount * trueBranchGini + node->m_falseSamplesCount * falseBranchGini) / (node->m_trueSamplesCount + node->m_falseSamplesCount);
#	else
	node->m_giniImpurity = (node->m_trueSamplesCumulatedWeightedVotes * trueBranchGini + node->m_falseSamplesCumulatedWeightedVotes * falseBranchGini) / (node->m_trueSamplesCumulatedWeightedVotes + node->m_falseSamplesCumulatedWeightedVotes);
#	endif
#else
	if (node->m_falseSamplesVector->size() != 0) {
		for (uint64_t i = 0; i < g_numClasses; i++) {
			double val = (double)node->m_falseClassesCount[i] / (double)node->m_falseSamplesVector->size();
			falseBranchGini -= val * val;
		}
	}
	node->m_giniImpurity = (node->m_trueSamplesVector->size() * trueBranchGini + node->m_falseSamplesVector->size() * falseBranchGini) / (node->m_trueSamplesVector->size() + node->m_falseSamplesVector->size());
#endif
	//LOG_DEBUG("Gini: %f %f\n", trueBranchGini, falseBranchGini);
}

	// At this point, the node inputSamples are filled, with data in order
void computeThreshold(CandidateNode **node) {
	//LOG_DEBUG("Compute threshold Input samples! depth: %lu #inputs: %lu\n", (*node)->m_featuresDepth, (*node)->m_inputSamplesCount);
	//LOG_DEBUG("Compute threshold Input samples! depth: %lu feature %lu #inputs: %lu\n", (g_numFeatures - (*node)->m_availableFeatures->size()), (*node)->m_featureIndex, (*node)->m_inputSamplesCount);
	SampleIndex *sIdPtr = (*node)->m_inputSamples;
	if (sIdPtr == NULL || (*node)->m_inputSamplesCount == 0) {
		LOG_CRITICAL("Cannot compute threshold, node does not contain ordered input!\n", 1);
		return;
	}
	double previous = 0.0;
	uint64_t count = 0;
	CandidateNode *chosen = NULL;
	double lowestGini = 1.0;
	double oldAvg = NAN;
	while (sIdPtr != NULL) {
		if (count > 0) {
			double avg = (sIdPtr->m_value + previous) / 2.0;
			if (avg != oldAvg) {
				oldAvg = avg;
#ifdef PARALLELIZE_GINI_THRESHOLDS
				cilk_spawn computeGiniThreshold_thread(g_trainSamples[sIdPtr->m_index], node, avg, count, &chosen, &lowestGini);
#else
				computeGiniThreshold_thread(g_trainSamples[sIdPtr->m_index], node, avg, count, &chosen, &lowestGini);
#endif
			} 
		}
		previous = sIdPtr->m_value;
		sIdPtr = sIdPtr->m_next;
		count++;
	}
#ifdef PARALLELIZE_GINI_THRESHOLDS
	cilk_sync;
#endif
	chosen->m_availableFeatures = new std::set<uint64_t>();
	for (std::set<uint64_t>::iterator it = (*node)->m_availableFeatures->begin(); it != (*node)->m_availableFeatures->end(); ++it) {
		if (*it == (*node)->m_featureIndex)
			continue;
		chosen->m_availableFeatures->insert(*it);
	}

	freeCandidateNode(*node);
	*node = chosen;
	//LOG_DEBUG("Feature index %lu Gini= %f threshold = %f\n", (*node)->m_featureIndex, (*node)->m_giniImpurity, (*node)->m_threshold);
} // void computeThreshold(CandidateNode **node)

CandidateNode *createChildNode(uint64_t tid, CandidateNode *parent, uint64_t featureIndex, bool branch) {
	CandidateNode *node = (CandidateNode *)mw_localmalloc(sizeof(CandidateNode), &g_trees[tid]);
	/*
	uint64_t numInputSamples = (branch ? parent->m_trueSamplesVector->size() : parent->m_falseSamplesVector->size());
	uint64_t *inputSamplesPtr = (branch ? parent->m_trueSamples : parent->m_falseSamples);

	for (uint64_t i = 0; i < numInputSamples; i++) {
		uint64_t s = inputSamplesPtr[i];
		SampleIndex *sample = (SampleIndex *)mw_localmalloc(sizeof(SampleIndex), node);
		sample->m_index = s;
		//cilk_spawn 
			insertSampleInput(g_samples[s], sample, featureIndex, node);
	}*/
	return node;
}

void computeGiniThreshold_thread(void *unused, CandidateNode **node, double avg, uint64_t count, CandidateNode **chosen, double *lowestGini) {
	//LOG_DEBUG("Compute threshold inside thread... %f\n", avg);
	CandidateNode *cNode = allocateCandidateNode(unused, (*node)->m_featureIndex, (*node)->m_inputSamplesCount, *node);
	//double gini = computeGiniThreshold(*node, avg, cNode);
	double gini = computeGiniThreshold(*node, avg, count, cNode);
#ifdef PARALLELIZE_GINI_THRESHOLDS
	while (true) {
		double oldGini = *lowestGini;
		if (gini > oldGini) {
			freeCandidateNode(cNode);
			return;
		}
		CandidateNode *oldChosen = *chosen;
		if (ATOMIC_CAS((long *)chosen, (long)cNode, (long)oldChosen) == (long)oldChosen) {
			*lowestGini = gini;
			if (oldChosen != NULL)
				freeCandidateNode(oldChosen);
			return;
		}
	}
#else
	if (gini > *lowestGini) {
		freeCandidateNode(cNode);
		return;
	}
	*lowestGini = gini;
	if (*chosen != NULL)
		freeCandidateNode(*chosen);
	*chosen = cNode;
#endif
}

double computeGiniThreshold(CandidateNode *node, double average, uint64_t averageIndex, CandidateNode *candidateNode) {
	//LOG_DEBUG("Computing Gini for feature %lu, average: %f [%lu]\n", candidateNode->m_featureIndex, average, averageIndex);
	candidateNode->m_threshold = average;

	SampleIndex *sptr = node->m_inputSamples; // Not going to copy the input to every candidate!
	uint64_t idx = 0;
	while(sptr != NULL) {
		uint64_t s = sptr->m_index;
		//double *sample = sptr->m_sample;
		//double val = g_samples[s][node->m_featureIndex];
		uint64_t label = g_trainSamplesLabels[s];
		// Note: we need to test if this is faster than passing the index in the linked list
		// Since we know that branch = index < index_"average + 1"
		//bool branch = (val < average); 
		bool branch = (idx < averageIndex);
#ifndef USE_BAGGING
		insertOutput(candidateNode, branch, s, label);
#else
		//LOG_DEBUG("insert output: %s, %lu %lu\n", (branch ? "true" : "false"), sptr->m_index, sptr->m_weightedVote);
		insertOutput(candidateNode, branch, s, label, sptr->m_weightedVote);
#endif
		//insertOutput_r(candidateNode, branch, s, label);
		//insertOutput_r(candidateNode, branch, sample, label);

		sptr = sptr->m_next;
		idx++;
	}
	//cilk_sync;
	computeGini(candidateNode);
	//LOG_DEBUG("result: %f\n", candidateNode->m_giniImpurity);
	return candidateNode->m_giniImpurity;
}

// Order based on featureIndex!
#ifndef USE_BAGGING
void fillInput(CandidateNode *parent, CandidateNode *child, bool branch) {
#else
void fillInput(CandidateNode *parent, CandidateNode *child, uint64_t tree, bool branch) {
#endif
	uint64_t featureIndex = child->m_featureIndex;
	//LOG_DEBUG("Fill input from @%p to @%p, index %lu! count: %lu @%p -> @%p\n", parent, child, featureIndex, samplesCount, samples, child->m_inputSamples); 
	//LOG_DEBUG("Fill input %s! @%p @%p\n", (branch ? "true" : "false"), parent->m_trueSamplesWeights, parent->m_falseSamplesWeights);
#ifndef VECTOR_OUTPUTS
//#	ifndef USE_BAGGING
	uint64_t samplesCount = (branch ? parent->m_trueSamplesCount : parent->m_falseSamplesCount);
//#	else
//	uint64_t samplesCount = (branch ? parent->m_trueSamplesCumulatedWeightedVotes : parent->m_falseSamplesCumulatedWeightedVotes);
//#	endif
	uint64_t *samples = (branch ? parent->m_trueSamples : parent->m_falseSamples);
	for (uint64_t i = 0; i < samplesCount; i++) {
		uint64_t s = samples[i];
		//LOG_DEBUG("copy input %s [%lu] s:%lu\n", (branch ? "true" : "false"), i, s);
#	ifndef USE_BAGGING
		insertInput(child, s, g_trainSamples[s][featureIndex]);
#	else
		uint64_t weight = g_samplesWeights[tree][s];
		insertInput(child, s, g_trainSamples[s][featureIndex], weight);
#	endif
	}
#else
	std::vector<uint64_t> *samples = (branch ? parent->m_trueSamplesVector : parent->m_falseSamplesVector);
	for (std::vector<uint64_t>::iterator it = samples->begin(); it != samples->end(); ++it) 
		insertInput(child, *it, g_trainSamples[*it][featureIndex]);
#endif	
	//LOG_DEBUG("Finished fill input for index %lu\n", featureIndex);
}

#ifndef USE_BAGGING
void insertInput(CandidateNode *node, uint64_t sampleIndex, double value) {
#else
void insertInput(CandidateNode *node, uint64_t sampleIndex, double value, uint64_t weight) {
#endif
//void insertInput(CandidateNode *node, double *samplePtr, double value) {
	node->m_inputSamplesCount++;
	SampleIndex *sPtr = node->m_inputSamples;
	SampleIndex *sample = (SampleIndex *)mw_localmalloc(sizeof(SampleIndex), (long *)node);
	sample->m_index = sampleIndex;
	//sample->m_sample = g_samples[sampleIndex];
	sample->m_value = value;
#ifdef USE_BAGGING
	sample->m_weightedVote = weight;
#endif
	//sample->m_previous = NULL;
	sample->m_next = NULL;
	if (sPtr == NULL) {
		node->m_inputSamples = sample;
		return;
	}
	SampleIndex *oldPtr = NULL;
	while (sPtr != NULL) {
		if (sPtr->m_value > sample->m_value)
			break;
		oldPtr = sPtr;
		sPtr = sPtr->m_next;
	}
	if (oldPtr != NULL) {
		//sample->m_previous = oldPtr;
		oldPtr->m_next = sample;
	} else
		node->m_inputSamples = sample;
	sample->m_next = sPtr;
}

#ifndef USE_BAGGING
void insertOutput(CandidateNode *node, bool branch, uint64_t sampleIndex, uint64_t label) {
#else
void insertOutput(CandidateNode *node, bool branch, uint64_t sampleIndex, uint64_t label, uint64_t weight) {
#endif
	//if (weight > 1)
	//	LOG_DEBUG("weight? %lu\n", weight);
#ifndef VECTOR_OUTPUTS
	uint64_t *count = (branch ? &node->m_trueSamplesCount : &node->m_falseSamplesCount);
#	ifdef USE_BAGGING
	uint64_t *votes = (branch ? &node->m_trueSamplesCumulatedWeightedVotes : &node->m_falseSamplesCumulatedWeightedVotes);
#	endif
	uint64_t countVal = *count;
	uint64_t *samples = (branch ? node->m_trueSamples : node->m_falseSamples);
	samples[countVal] = sampleIndex;
#ifdef USE_BAGGING
	REMOTE_ADD((long *)votes, weight);
#endif
	REMOTE_ADD((long *)count, 1);
#else
	std::vector<uint64_t> *samplesVector = (branch ? node->m_trueSamplesVector : node->m_falseSamplesVector);
	samplesVector->push_back(sampleIndex);
#endif


#ifndef USE_BAGGING
	uint64_t *samplesClasses = (branch ? node->m_trueClassesCount : node->m_falseClassesCount);
	REMOTE_ADD((long *)&samplesClasses[label], 1);
#else
	uint64_t *samplesClasses = (branch ? node->m_trueClassesWeightedVotes : node->m_falseClassesWeightedVotes);
	REMOTE_ADD((long *)&samplesClasses[label], weight);
#endif
	//LOG_DEBUG("After test: %s %lu vs: %lu\n", (branch ? "true" : "false"), (branch ? node->m_trueSamplesVector->size() : node->m_falseSamplesVector->size()), *count); 
}

/*
void insertOutput_r(CandidateNode *node, bool branch, uint64_t sampleIndex, uint64_t label) {
	//uint64_t *count = (branch ? &node->m_trueSamplesVector->size() : &node->m_falseSamplesVector->size());

	//uint64_t countVal = *count;
	//uint64_t *samples = (branch ? node->m_trueSamples : node->m_falseSamples);
	std::vector<uint64_t> *samples = (branch ? node->m_trueSamplesVector : node->m_falseSamplesVector);
	uint64_t *samplesClasses = (branch ? node->m_trueClassesCount : node->m_falseClassesCount);
	//LOG_DEBUG("Inserting in output: sample %lu, label: %lu, test: %s @%p @%p @%p\n", sampleIndex, label, (branch ? "true" : "false"), node, samples, samplesClasses);
	//while (ATOMIC_CAS((long *)count, countVal + 1, countVal) != countVal) { 
	//	countVal++;
	//}
	//LOG_DEBUG("appending at position %lu index %lu branch %s\n", *count, sampleIndex, (branch ? "true" : "false"));
	//samples[*count - 1] = sampleIndex;
	samples->push_back(sampleIndex);
	ATOMIC_ADDM((long *)&samplesClasses[label], 1);
	//LOG_DEBUG("After test: %s %lu vs: %lu\n", (branch ? "true" : "false"), (branch ? node->m_trueSamplesVector->size() : node->m_falseSamplesVector->size()), *count); 
}
*/

// DEBUG functions
void printInput(CandidateNode *node) {
	SampleIndex *sptr = node->m_inputSamples;
	while (sptr != NULL) {
		LOG_DEBUG(">> %03lu, %f\n", sptr->m_index, sptr->m_value);
		//LOG_DEBUG(">> %p, %f\n", sptr->m_sample, sptr->m_value);
		sptr = sptr->m_next;
	}
}

void applyTree(uint64_t sampleId, Node *rootNode) {
	Node *node = rootNode;
	//LOG_DEBUG("@%p\n", node);
	//LOG_DEBUG("@%p @%p @%p\n", node->m_trueBranch, node->m_falseBranch);
	//while (node->m_trueBranch != NULL && node->m_falseBranch != NULL) {
	while (node->m_labelsVotes == NULL) {
		//LOG_DEBUG("Node: @%p val[%lu] < %f\n", node, node->m_featureIndex, node->m_threshold);
#ifdef INVERT_TEST_DATA
		//bool b = (g_testSamples[node->m_featureIndex][sampleId] < node->m_threshold);
		node = (g_testSamples[node->m_featureIndex][sampleId] < node->m_threshold ? node->m_trueBranch : node->m_falseBranch);
#else
		//bool b = (g_testSamples[sampleId][node->m_featureIndex] < node->m_threshold);
		node = (g_testSamples[sampleId][node->m_featureIndex] < node->m_threshold ? node->m_trueBranch : node->m_falseBranch);
#endif
		//LOG_DEBUG("[%s] next node? @%p\n", (b ? "true" : "false"), node);
	}
	/*
	std::stringstream ss;
	ss << "[ ";
	for (uint64_t i = 0; i < g_numFeatures; i++) {
#ifdef INVERT_TEST_DATA
		ss << g_testSamples[i][sampleId] << " ";
#else
		ss << g_testSamples[sampleId][i] << " ";
#endif
	}
	ss << "]";
	LOG_DEBUG("sample %lu %s, class: %lu\n", sampleId, ss.str().c_str(), node->m_featureIndex);
	*/
	//REMOTE_ADD((long *)&g_testSamplesClassesVotes[sampleId][node->m_featureIndex], 1);
	//LOG_DEBUG("labels? @%p-> @%p\n", node, node->m_labelsVotes);
	for (uint64_t i = 0; i < g_numClasses; i++) {
		//LOG_DEBUG("tree label[%lu]= %lu votes update global @%p\n", i, node->m_labelsVotes[i], g_testSamplesLabelsVotes[sampleId]);
		ATOMIC_ADDM((long *)&g_testSamplesLabelsVotes[sampleId][i], node->m_labelsVotes[i]);
	}
	//LOG_DEBUG("@%p\n", rootNode);
}

void loopApplyTreeRM(uint64_t sampleId, uint64_t treeNum)
	{	// run applyTreeRM in a loop, to test scaling
	for (uint64_t t = treeNum; t < treeNum + g_treesPerThread; t++)
		applyTreeRM(sampleId, g_trees[t]);
	}	// loopApplyTreeRM()

void applyTreeRM(uint64_t sampleId, Node *rootNode) {
	Node *node = rootNode;
	//LOG_DEBUG("@%p\n", node);
	//LOG_DEBUG("@%p @%p @%p\n", node->m_trueBranch, node->m_falseBranch);
	while (!node->_isLeaf)
		{
		//LOG_DEBUG("Node: @%p ind[%lu] < %f\n", node, node->m_featureIndex, node->m_threshold);
		//LOG_DEBUG("Node: @%p val[%d] < %f\n", node, g_testSamples[sampleId][node->m_featureIndex], node->m_threshold);
		//bool b = (g_testSamples[sampleId][node->m_featureIndex] < node->m_threshold);
		node = (g_testSamples[sampleId][node->m_featureIndex] < node->m_threshold ? node->m_trueBranch : node->m_falseBranch);
		//LOG_DEBUG("[%s] next node? @%p\n", (b ? "true" : "false"), node);
		}
	//LOG_DEBUG("Leaf prob for sample %d, node @%p is %f, longed is %d\n", sampleId, node, node->_prob, int(node->_prob * 10000));
	//REMOTE_ADD((double *)&g_testSamplesClassesProbs[node->_class], node->_prob);
	REMOTE_ADD((long *)&g_testSamplesClassesProbs[sampleId][node->_class], int(node->_prob * 10000));
}

void test() {
	test_spawnRecursive(0, NODELETS(), g_trainSamples[0]);
	//for (uint64_t k = 0; k < g_numTestSamples; k++)
	//	testSample(k);
}

void testRM() {
  volatile uint64_t start = CLOCK();
	//applyTreeRM(0, g_trees[0]);
	//applyTreeRM(0, g_trees[1]);
	//cilk_spawn testSampleRM(0);
	//cilk_spawn testSampleRM(1);
	//cilk_sync;
	//cilk_spawn testSampleRM(2);
	test_spawnRecursiveRM(0, NODELETS(), g_testSamples[0]);
	//for (uint64_t k = 0; k < g_numTestSamples; k++)
	//	testSampleRM(k);
	volatile uint64_t stop = CLOCK();
	long testCycles = stop - start;
	LOG_DEBUG( "Testing took %d cycles\n", testCycles );
}

void test_spawnRecursive(uint64_t low, uint64_t high, void *dummy) {
    while(true) {
        uint64_t count = high - low;
        if (count <= 1)
            break;
        uint64_t mid = low + count / 2;
        cilk_spawn test_spawnRecursive(mid, high, g_trainSamples[mid]);
        high = mid;
    }   

    for (uint64_t k = low; k < g_numTestSamples; k += NODELETS()) 
        //testSample(k);
        cilk_spawn testSample(k);
}

void test_spawnRecursiveRM(uint64_t low, uint64_t high, void *dummy) {
    while(true) {
        uint64_t count = high - low;
        if (count <= 1)
            break;
        uint64_t mid = low + count / 2;
        cilk_spawn test_spawnRecursiveRM(mid, high, g_testSamples[mid]);
        high = mid;
    }   

	// now each nodelet has a single thread
	// each of those nodelet threads handles the samples on its nodelet
	// by spawning a thread per sample
    //for (uint64_t k = low; k < g_numTestSamples; k += NODELETS()) 
    for (uint64_t k = low; k < g_sampleMax; k += NODELETS()) 
        //testSample(k);
      /*cilk_spawn*/ testSampleRM(k);
	// implicit cilk_sync on return
}

void testSample(uint64_t sampleId) {
  //LOG_DEBUG("Classifying sample %lu\n", sampleId);
	for (uint64_t t = 0; t < g_numTrees; t++) {
		//LOG_DEBUG("Tree #%lu\n", t);
		cilk_spawn applyTree(sampleId, g_trees[t]);
	}
	cilk_sync;
	int64_t label = -1;
	uint64_t votes = 0;
	for (uint64_t i = 0; i < g_numClasses; i++) {
		//LOG_DEBUG("labels votes?? [%lu] [%lu] -> %lu\n", sampleId, i, g_testSamplesLabelsVotes[sampleId][i]);
		if (g_testSamplesLabelsVotes[sampleId][i] > votes) {
			votes = g_testSamplesLabelsVotes[sampleId][i];
			label = i;
		}
	}
	/*
	LOG_DEBUG("Sample %lu class: %lu/%lu? %s\n", sampleId, label, g_testSamplesLabels[sampleId], (label == g_testSamplesLabels[sampleId] ? "true" : "false"));
	if (label == g_testSamplesLabels[sampleId])
		ATOMIC_ADDM((long *)&g_rightResults, 1);
	else
		ATOMIC_ADDM((long *)&g_wrongResults, 1);
		*/
	//LOG_DEBUG("sample %lu classified as %ld\n", sampleId, label);
	g_testSamplesLabels[sampleId] = label;

}
void testSampleRM(uint64_t sampleId) {
	// each sample thread spawns a thread per tree
	//LOG_DEBUG("Classifying sample %lu\n", sampleId);
	//for (uint64_t t = 0; t < g_numTrees; t++) {
#if 0
	for (uint64_t t = 0; t < g_treeMax; t++) {
		//LOG_DEBUG("Applying tree #%lu for sample #%lu\n", t, sampleId);
		cilk_spawn applyTreeRM(sampleId, g_trees[t]);
		//applyTreeRM(sampleId, g_trees[t]);
	}
#else
	//LOG_DEBUG( "Spawned thread for sample %d\n", sampleId );
	for (uint64_t t = 0; t < g_treeMax; t+=g_treesPerThread)
		{
		cilk_spawn loopApplyTreeRM(sampleId,t);
		}
#endif
	cilk_sync;
	int64_t label = -1;
	uint64_t prob = 0;
	for (uint64_t i = 0; i < g_numClasses; i++) {
		//LOG_DEBUG("labels votes?? [%lu] [%lu] -> %lu\n", sampleId, i, g_testSamplesLabelsVotes[sampleId][i]);
		if (g_testSamplesClassesProbs[sampleId][i] > prob) {
			prob = g_testSamplesClassesProbs[sampleId][i];
			label = i;
		}
	}
	//if (sampleId < 5)
	//	LOG_DEBUG("sample %lu classified as %ld\n", sampleId, label);
	g_testSamplesLabels[sampleId] = label;
}

//
// rfc_api implementation
//
extern "C" {

void randomForestClassifier_initialize(uint64_t numTrees, MaxFeatures maxFeatures, uint64_t randomState) {
	mw_replicated_init((long *)&g_numTrees, (long)numTrees);
	g_trainingState.m_maxFeatures = maxFeatures;
	srand(randomState);
	initTrees();
}

// initialization for the read model path
// set up distributed array that will point to trees

void randomForestClassifier_RMinitialize(uint64_t numTrees, uint64_t treeMax, uint64_t treesPerThread) {
	mw_replicated_init((long *)&g_numTrees, (long)numTrees);
	// limit the testing to this number of trees:
	if ((treeMax < 1) || (treeMax > numTrees))
		treeMax = numTrees;		// keep this in bounds
	mw_replicated_init((long *)&g_treeMax, treeMax);
	if ((treesPerThread < 1) || (treesPerThread > treeMax))
		treesPerThread = 1;	// keep in bounds
	mw_replicated_init((long *)&g_treesPerThread, treesPerThread);
	initTrees();
}

void randomForestClassifier_initializeTraining(uint64_t numSamples, uint64_t numFeatures, uint64_t numClasses) {
	mw_replicated_init((long *)&g_numTrainSamples, (long)numSamples);
	mw_replicated_init((long *)&g_numFeatures, (long)numFeatures);
	mw_replicated_init((long *)&g_numClasses, (long)numClasses);

	double **samplesWeightsTmp = (double **)mw_malloc2d(g_numTrees, numSamples * sizeof(double));
	mw_replicated_init((long *)&g_samplesWeights, (long)samplesWeightsTmp);

	uint64_t maxFeatures = numFeatures;
	switch (g_trainingState.m_maxFeatures) {
	case AUTO:
	case SQRT:
		maxFeatures = sqrt(numFeatures);
		break;
	case LOG2:
		maxFeatures = log2(numFeatures);
		break;
	default:
		;
	}
	mw_replicated_init((long *)&g_selectionSize, (long)maxFeatures);

	double **samplesTrainTmp = (double **)mw_malloc2d(numSamples, numFeatures * sizeof(double));
	mw_replicated_init((long *)&g_trainSamples, (long)samplesTrainTmp);

	uint64_t *samplesLabelsTrainTmp = (uint64_t *)mw_malloc1dlong(numSamples);
	mw_replicated_init((long *)&g_trainSamplesLabels, (long)samplesLabelsTrainTmp);
}

void randomForestClassifier_fillTrainingData(double *gc_samples, uint64_t *gc_classes) {
	LOG_DEBUG("Copy data from samples buffer to distributed 2D array\n");
	for (uint64_t s = 0; s < g_numTrainSamples; s++) {
		for (uint64_t f = 0; f < g_numFeatures; f++)
			g_trainSamples[s][f] = gc_samples[s * g_numFeatures + f];
	}
	LOG_DEBUG("Copy data from classes buffer to distributed 1D array\n");
	for (uint64_t s = 0; s < g_numTrainSamples; s++) {
		g_trainSamplesLabels[s] = gc_classes[s];
	}
	LOG_DEBUG("Training data copy finished\n");
}

void randomForestClassifier_train() {
	train();
}

void randomForestClassifier_readModel(uint64_t t, int nodeNum, FILE* modelFile) {
	readTree(t,nodeNum,modelFile);
}

void randomForestClassifier_initializeTesting(uint64_t numSamples) {
	LOG_DEBUG("Initialize testing with %lu samples (%lu classes)\n", numSamples, g_numClasses);
	mw_replicated_init((long *)&g_numTestSamples, (long)numSamples);
#ifdef INVERT_TEST_DATA
	double **samplesTestTmp = (double **)mw_malloc2d(g_numFeatures, numSamples * sizeof(double));
#else
	double **samplesTestTmp = (double **)mw_malloc2d(numSamples, g_numFeatures * sizeof(double));
#endif
	mw_replicated_init((long *)&g_testSamples, (long)samplesTestTmp);

	uint64_t *samplesLabelsTestTmp = (uint64_t *)mw_malloc1dlong(numSamples);
	for (uint64_t i = 0; i < numSamples; i++)
		samplesLabelsTestTmp[i] = -1L;	// init these to prevent problems
	mw_replicated_init((long *)&g_testSamplesLabels, (long)samplesLabelsTestTmp);

	uint64_t **samplesTestLabelsTmp = (uint64_t **)mw_malloc2d(numSamples, g_numClasses * sizeof(uint64_t));
	mw_replicated_init((long *)&g_testSamplesLabelsVotes, (long)samplesTestLabelsTmp);

	for (uint64_t s = 0; s < numSamples; s++) {
		for (uint64_t i = 0; i < g_numClasses; i++)
			g_testSamplesLabelsVotes[s][i] = 0;
	}
}

void randomForestClassifier_RMinitializeTesting(uint64_t numClasses, uint64_t numFeatures, uint64_t numSamples, uint64_t sampleMax) {
	mw_replicated_init((long *)&g_numClasses, (long)numClasses);
	mw_replicated_init((long *)&g_numFeatures, (long)numFeatures);
	mw_replicated_init((long *)&g_numTestSamples, (long)numSamples);
	if ((sampleMax < 1) || (sampleMax > numSamples))
		sampleMax = numSamples;		// keep this in bounds
	mw_replicated_init((long *)&g_sampleMax, sampleMax);
	LOG_DEBUG("Initialize testing with %lu samples (%lu features)\n", sampleMax, numFeatures);
	LOG_DEBUG( "Testing samples: %d of %d against trees: %d of %d\n", g_sampleMax, g_numTestSamples, g_treeMax, g_numTrees );
	// TODO:  maybe this should be "inverted"
	double **samplesTestTmp = (double **)mw_malloc2d(numSamples, g_numFeatures * sizeof(double));
	mw_replicated_init((long *)&g_testSamples, (long)samplesTestTmp);

	uint64_t *samplesLabelsTestTmp = (uint64_t *)mw_malloc1dlong(numSamples);
	mw_replicated_init((long *)&g_testSamplesLabels, (long)samplesLabelsTestTmp);

	uint64_t **samplesTestClassesTmp = (uint64_t **)mw_malloc2d(numSamples, g_numClasses * sizeof(uint64_t));
	mw_replicated_init((long *)&g_testSamplesClassesProbs, (long)samplesTestClassesTmp);
	for (uint64_t s = 0; s < numSamples; s++) {
		for (uint64_t i = 0; i < g_numClasses; i++)
			g_testSamplesClassesProbs[s][i] = 0;
	}
}

void randomForestClassifier_fillTestingData(double *gc_samples) {
	LOG_DEBUG("Copy data from test samples buffer to distributed 2D array\n");
	for (uint64_t s = 0; s < g_numTestSamples; s++) {
		for (uint64_t f = 0; f < g_numFeatures; f++) {
			g_testSamples[s][f] = gc_samples[s * g_numFeatures + f];
		}
	}
	printf( "g_numFeatures, g_numTestSamples is %d, %d\n", g_numFeatures, g_numTestSamples );
}

void randomForestClassifier_RMfillTestingData(double *gc_samples) {
	LOG_DEBUG("Copy data from test samples buffer to distributed 2D array\n");
	for (uint64_t s = 0; s < g_numTestSamples; s++) {
		for (uint64_t f = 0; f < g_numFeatures; f++) {
			g_testSamples[s][f] = gc_samples[s * g_numFeatures + f];
		}
	}
	printf( "g_numFeatures, g_numTestSamples is %d, %d\n", g_numFeatures, g_numTestSamples );
}

void randomForestClassifier_test(uint64_t *gc_classified) {
	testRM();
	for (uint64_t s = 0; s < g_numTestSamples; s++) {
		gc_classified[s] = g_testSamplesLabels[s];
	}
#if 0
	test();
	for (uint64_t s = 0; s < g_numTestSamples; s++) {
		gc_classified[s] = g_testSamplesLabels[s];
	}
#endif
	}

} // extern "C"

