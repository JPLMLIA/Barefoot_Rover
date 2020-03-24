//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: decision_tree.h
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Structures and prototypes
// ========================================================================
#pragma once
#include <stdint.h>

typedef struct node_t Node;

struct node_t {
	uint64_t  m_featureIndex;
	double    m_threshold; // computed from samplesIndex, average values on the feature 
	
	double    m_giniImpurity;

	Node   *m_trueBranch;	
	Node   *m_falseBranch;	

	// This is used on leafs only
	uint64_t *m_labelsVotes; // [numClasses]

	// for readModel
	uint64_t _isLeaf;
	uint64_t _class;
	double   _prob;
};

// Apply the tree to a sample, stored as an array of doubles
void applyTree(uint64_t sampleId, Node *);
void applyTreeRM(uint64_t sampleId, Node *);

void test_spawnRecursive(uint64_t, uint64_t, void *); 
void test_spawnRecursiveRM(uint64_t, uint64_t, void *); 
void testSample(uint64_t);
void testSampleRM(uint64_t);

