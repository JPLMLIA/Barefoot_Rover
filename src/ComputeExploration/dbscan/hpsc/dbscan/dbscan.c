/* Copyright 2015 Gagarine Yaikhom (MIT License) */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNCLASSIFIED -1
#define NOISE -2

#define CORE_POINT 1
#define NOT_CORE_POINT 0

#define SUCCESS 0
#define FAILURE -3

typedef struct point_s point_t;
struct point_s {
    double features[3];
};

typedef struct node_s node_t;
struct node_s {
    unsigned int index;
    node_t *next;
};

typedef struct epsilon_neighbours_s epsilon_neighbours_t;
struct epsilon_neighbours_s {
    unsigned int num_members;
    node_t *head, *tail;
};

node_t *create_node(unsigned int index);
int append_at_end(
     unsigned int index,
     epsilon_neighbours_t *en);
epsilon_neighbours_t *get_epsilon_neighbours(
    unsigned int index,
	unsigned int num_features,
    double points[][num_features],
    unsigned int num_points,
    double epsilon,
    double (*dist)(double *a, double *b));
void print_epsilon_neighbours(
    point_t *points,
    epsilon_neighbours_t *en);
void destroy_epsilon_neighbours(epsilon_neighbours_t *en);
void dbscan(
	unsigned int num_features,
	double points[][num_features],
	int labels[],
    unsigned int num_points,
    double epsilon,
    unsigned int minpts,
    double (*dist)(double *a, double *b));
int expand(
    unsigned int index,
    unsigned int cluster_id,
	unsigned int num_features,
    double points[][num_features],
	int labels[],
    unsigned int num_points,
    double epsilon,
    unsigned int minpts,
    double (*dist)(double *a, double *b));
int spread(
    unsigned int index,
    epsilon_neighbours_t *seeds,
    unsigned int cluster_id,
	unsigned int num_features,
    double points[][num_features],
	int labels[],
    unsigned int num_points,
    double epsilon,
    unsigned int minpts,
    double (*dist)(double *a, double *b));
double euclidean_dist(double *a, double *b);
double adjacent_intensity_dist(point_t *a, point_t *b);
unsigned int parse_input1(
    FILE *file,
    double *epsilon,
    unsigned int *minpts,
	unsigned int *num_features);
void parse_input2(
    FILE *file2,
	unsigned int num_features,
    double p[][num_features],
	int labels[],
	unsigned int num_points);
void print_points(
	unsigned int num_features,
	double points[][num_features],
	int labels[],
    unsigned int num_points);

FILE	*file2;
unsigned int num_features;

node_t *create_node(unsigned int index)
{
    node_t *n = (node_t *) calloc(1, sizeof(node_t));
    if (n == NULL)
        perror("Failed to allocate node.");
    else {
        n->index = index;
        n->next = NULL;
    }
    return n;
}

int append_at_end(
     unsigned int index,
     epsilon_neighbours_t *en)
{
    node_t *n = create_node(index);
    if (n == NULL) {
        free(en);
        return FAILURE;
    }
    if (en->head == NULL) {
        en->head = n;
        en->tail = n;
    } else {
        en->tail->next = n;
        en->tail = n;
    }
    ++(en->num_members);
    return SUCCESS;
}

epsilon_neighbours_t *get_epsilon_neighbours(
    unsigned int index,
	unsigned int num_features,
    double points[][num_features],
    unsigned int num_points,
    double epsilon,
    double (*dist)(double *a, double *b))
{
    epsilon_neighbours_t *en = (epsilon_neighbours_t *)
        calloc(1, sizeof(epsilon_neighbours_t));
    if (en == NULL) {
        perror("Failed to allocate epsilon neighbours.");
        return en;
    }
    for (int i = 0; i < num_points; ++i) {
        if (i == index)
            continue;
        if (dist(points[index], points[i]) > epsilon)
            continue;
        else {
            if (append_at_end(i, en) == FAILURE) {
                destroy_epsilon_neighbours(en);
                en = NULL;
                break;
            }
        }
    }
    return en;
}

void print_epsilon_neighbours(
    point_t *points,
    epsilon_neighbours_t *en)
{
    if (en) {
        node_t *h = en->head;
        while (h) {
            printf("(%lfm, %lf, %lf)\n",
                   points[h->index].features[0],
                   points[h->index].features[1],
                   points[h->index].features[2]);
            h = h->next;
        }
    }
}

void destroy_epsilon_neighbours(epsilon_neighbours_t *en)
{
    if (en) {
        node_t *t, *h = en->head;
        while (h) {
            t = h->next;
            free(h);
            h = t;
        }
        free(en);
    }
}

void dbscan(
	unsigned int num_features,
    double points[][num_features],
	int labels[],
    unsigned int num_points,
    double epsilon,
    unsigned int minpts,
    double (*dist)(double *a, double *b))
{
    unsigned int i, cluster_id = 0;
    for (i = 0; i < num_points; ++i) {
        if (labels[i] == UNCLASSIFIED) {
            if (expand(i, cluster_id, num_features, points, labels,
                       num_points, epsilon, minpts,
                       dist) == CORE_POINT)
                ++cluster_id;
        }
    }
}

int expand(
    unsigned int index,
    unsigned int cluster_id,
	unsigned int num_features,
    double points[][num_features],
	int labels[],
    unsigned int num_points,
    double epsilon,
    unsigned int minpts,
    double (*dist)(double *a, double *b))
{
    int return_value = NOT_CORE_POINT;
    epsilon_neighbours_t *seeds =
        get_epsilon_neighbours(index, num_features, points,
                               num_points, epsilon,
                               dist);
    if (seeds == NULL)
        return FAILURE;

    if (seeds->num_members < minpts)
        labels[index] = NOISE;
    else {
        labels[index] = cluster_id;
        node_t *h = seeds->head;
        while (h) {
            labels[h->index] = cluster_id;
            h = h->next;
        }

        h = seeds->head;
        while (h) {
            spread(h->index, seeds, cluster_id, num_features, points, labels,
                   num_points, epsilon, minpts, dist);
            h = h->next;
        }

        return_value = CORE_POINT;
    }
    destroy_epsilon_neighbours(seeds);
    return return_value;
}

int spread(
    unsigned int index,
    epsilon_neighbours_t *seeds,
    unsigned int cluster_id,
	unsigned int num_features,
    double points[][num_features],
	int labels[],
    unsigned int num_points,
    double epsilon,
    unsigned int minpts,
    double (*dist)(double *a, double *b))
{
    epsilon_neighbours_t *spread =
        get_epsilon_neighbours(index, num_features, points,
                       num_points, epsilon,
                       dist);
    if (spread == NULL)
        return FAILURE;
    if (spread->num_members >= minpts) {
        node_t *n = spread->head;
        while (n) {
			if (labels[n->index] == NOISE ||
					labels[n->index] == UNCLASSIFIED) {
				if (labels[n->index] == UNCLASSIFIED) {
					if (append_at_end(n->index, seeds)
						== FAILURE) {
						destroy_epsilon_neighbours(spread);
						return FAILURE;
					}
				}
				labels[n->index] = cluster_id;
            }
            n = n->next;
        }
    }

    destroy_epsilon_neighbours(spread);
    return SUCCESS;
}

double euclidean_dist(double *a, double *b)
{
	double sumsquares = 0;

	for (int i = 0; i < num_features; i++)
		sumsquares += (pow(a[i] - b[i], 2));
    return sqrt(sumsquares);
}

unsigned int parse_input1(
    FILE *file,
    double *epsilon,
    unsigned int *minpts,
	unsigned int *num_features)
{
    unsigned int num_points = 0;
    file2 = fopen("./example2.dat", "r");
    fscanf(file2, "%lf %u %u %u\n",
           epsilon, minpts, &num_points, num_features);
    printf("Number of features is %d\n", *num_features);
    return num_points;
}

void parse_input2(
    FILE *file2,
	unsigned int num_features,
    double p[][num_features],
	int labels[],
	unsigned int num_points)
{
    unsigned int i = 0;
    while (i < num_points) {
    	for (int j = 0; j < num_features; j++)
          fscanf(file2, "%lf",
                 &(p[i][j]));
        labels[i] = UNCLASSIFIED;
        ++i;
    }
}

void print_points(
	unsigned int num_features,
	double points[][num_features],
	int labels[],
    unsigned int num_points)
{
    unsigned int i = 0;
    unsigned int j;
    int features_count = num_features;
    if (features_count > 5)
    	features_count = 5;
    printf("Number of points: %u\n"
        " 0     1     2     cluster_id\n"
        "-----------------------------\n"
        , num_points);
    while (i < num_points) {
    	j = 0;
    	while (j < features_count) {
          printf("%5.2lf ", points[i][j]);
          ++j;
    	}
        printf(": %d\n", labels[i]);
          ++i;
    }
}

int main(void) {
    double epsilon;
    unsigned int minpts;
    unsigned int num_points =
        parse_input1(stdin, &epsilon, &minpts, &num_features);
    double points[num_points][num_features];
    int labels[num_points];
    if (num_points) {
    	parse_input2(file2, num_features, points, labels, num_points);
    }
    if (num_points) {
        dbscan(num_features, points, labels, num_points, epsilon,
               minpts, euclidean_dist);
        printf("Epsilon: %lf\n", epsilon);
        printf("Minimum points: %u\n", minpts);
        print_points(num_features, points, labels, num_points);
    }
    return 0;
}
