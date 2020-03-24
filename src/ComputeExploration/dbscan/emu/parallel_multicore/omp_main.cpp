/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   Files: omp_main.cpp clusters.cpp  clusters.h utils.h utils.cpp          */
/*   			dbscan.cpp dbscan.h kdtree2.cpp kdtree2.hpp          */
/*		    						             */
/*   Description: an openmp implementation of dbscan clustering algorithm    */
/*				using the disjoint set data structure        */
/*                                                                           */
/*   Author:  Md. Mostofa Ali Patwary                                        */
/*            EECS Department, Northwestern University                       */
/*            email: mpatwary@eecs.northwestern.edu                          */
/*                                                                           */
/*   Copyright, 2012, Northwestern University                                */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/*   Please cite the following publication if you use this package 	     */
/* 									     */
/*   Md. Mostofa Ali Patwary, Diana Palsetia, Ankit Agrawal, Wei-keng Liao,  */
/*   Fredrik Manne, and Alok Choudhary, "A New Scalable Parallel DBSCAN      */
/*   Algorithm Using the Disjoint Set Data Structure", Proceedings of the    */
/*   International Conference on High Performance Computing, Networking,     */
/*   Storage and Analysis (Supercomputing, SC'12), pp.62:1-62:11, 2012.	     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "dbscan.h"
#include "utils.h"
#include "kdtree2.hpp"
#include "memoryweb.h"

static void usage(char *argv0) 
{
    const char *params =
	"Usage: %s [switches] -i filename -b -m minpts -e epsilon -o output -t threads\n"
    	"	-i filename	: file containing input data to be clustered\n"
    	"	-b		: input file is in binary format (default no)\n"
	"	-m minpts	: input parameter of DBSCAN, min points to form a cluster, e.g. 2\n"
	"	-e epsilon	: input parameter of DBSCAN, radius or threshold on neighbourhoods retrieved, e.g. 0.8\n"
	"	-s bucketsize	: size of a leaf in the kd-tree, e.g. 12\n"
	"	-o output	: clustering results, format, (each line, point id, clusterid)\n"
	"	-t threads	: number of threads to be employed\n\n"
        "       -r replicated nodes : number of kdtree nodes to replicate\n\n";
		
    fprintf(stderr, params, argv0);
    exit(-1);
}

replicated long int NUM_REPLICATED_NODES;

int main(int argc, char** argv)
{
	double	seconds;
	int 	opt;

	int 	minPts, threads;
	int classical = 0;
	double 	eps;
	char* 	outfilename = NULL;
	int     isBinaryFile;
	char*   infilename = NULL;
	int     bucketsize;
	int	vectorCount = 0;	// 0 => use all vectors
	
	// some default values
	minPts 		= -1;
	eps		= -1;
	isBinaryFile 	= 0;
	outfilename 	= NULL;
	infilename	= NULL;
	threads 	= -1;
	bucketsize      = -1;
	
    	while ((opt=getopt(argc,argv,"f:i:t:r:d:p:m:e:o:s:v:z:bxghncul"))!= EOF)
    	{
		switch (opt)
        	{
			case 'f':
				vectorCount = atoi(optarg);
				break;
        		case 'i':
            			infilename = optarg;
               			break;
			case 't':
				threads = atoi(optarg);
				break;
		        case 'r':
                              	mw_replicated_init(&NUM_REPLICATED_NODES, atol(optarg));
	    		case 'b':
		            	isBinaryFile = 1;
            			break;
           		case 'm':
		            	minPts = atoi(optarg);
            			break;
		        case 'e':
            			eps  = atof(optarg);
            			break;
		        case 's':
			        bucketsize = atoi(optarg);
			        break;
			case 'o':
				outfilename = optarg;
				break;
			case 'c':
				classical = 1;
				break;
            		case '?':
           			usage(argv[0]);
	           		break;
           		default:
           			usage(argv[0]);
           			break;
    		}
	}

	if(infilename == NULL || minPts < 0 || eps < 0 || threads < 1 || bucketsize < 1)
	{
		usage(argv[0]);
		exit(-1);
	}

#if 0
	omp_set_num_threads(threads);
#endif

	NWUClustering::ClusteringAlgo dbs;
	dbs.set_dbscan_params(eps, minPts);

	cout << "Input parameters " << " minPts " << minPts << " eps " << eps << endl;

#if 0
	double start = omp_get_wtime();
#else
	volatile uint64_t start = CLOCK();
#endif
	cout << "Reading points from file: " << infilename << endl;
	if(dbs.read_file(infilename, isBinaryFile, vectorCount) == -1)
			exit(-1);
#if 0
	cout << "Reading input data file took " << omp_get_wtime() - start << " seconds." << endl;
#else
	cout << "Reading input data file took " << CLOCK() - start << " cycles." << endl;
#endif

	// build kdtree for the points
#if 0
	start = omp_get_wtime();
#else
	start = CLOCK();
#endif
	cout << "Bucket size = " << bucketsize << std::endl;
	dbs.build_kdtree(bucketsize);
#if 0
	cout << "Build kdtree took " << omp_get_wtime() - start << " seconds." << endl;
#else
	cout << "Build kdtree took " << CLOCK() - start << " cycles." << endl;
	//cout << "Early exit" << endl;
	//exit(8);
#endif

#if 0
	start = omp_get_wtime();
#endif
	//run_dbscan_algo(dbs);
	run_dbscan_algo_uf(dbs, threads);
#if 0
	cout << "DBSCAN (total) took " << omp_get_wtime() - start << " seconds." << endl;
#else
	cout << "DBSCAN (total) took " << CLOCK() - start << " cycles." << endl;
#endif

	if(outfilename != NULL)
	{
		ofstream outfile;
		outfile.open(outfilename);
		dbs.writeClusters_uf(outfile);
		//dbs.writeClusters(outfile);
		outfile.close();
	}
	
	exit(0);
}
