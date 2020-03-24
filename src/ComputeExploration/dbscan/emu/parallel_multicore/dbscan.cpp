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


#include "log.h"
#include "dbscan.h"
#include "globals.h"

// Emu data map:
// dbs:  	all on default nodelet (0?)
// merge:	distributed array of vectors, maxthread in size
// m_pts:	distributed
// ind:		on default nodelet
// prID:	on default nodelet

extern "C" {
#include <cilk.h>
#include <memoryweb.h>
#include <emu_c_utils/hooks.h>
}


#if REPL_KDTREE
extern replicated kdtree2 * m_kdtree;
#endif


namespace NWUClustering
{
	replicated long*	g_m_parents;
	replicated long*	g_m_corepoint;
	replicated long*	g_m_member;
	replicated vector <int> **merge;

	long		distMerges = 0;		// count of distributed merge
	

    	void ClusteringAlgo::set_dbscan_params(double eps, int minPts)
	{
		m_epsSquare =  eps * eps;
		m_minPts =  minPts;
	}

	ClusteringAlgo::~ClusteringAlgo()
	{
		m_noise.clear();
		m_visited.clear();
// should we mw_free()?  RandomForest code doesn't
#if 0
		m_parents.clear();

#endif
		mw_free(g_m_parents);
		mw_free(merge);
		mw_free(g_m_corepoint);
		mw_free(g_m_member);

		
	}
	
	void ClusteringAlgo::writeClusters(ostream& o)
	{
		// writing point id and cluster id pairs per line, noise has cluster id 0	
		int iMaxID = m_clusters.size(), id, i, j;
		for(i = 0; i < m_pts->m_i_num_points; i++)
		{
			//for (j = 0; j < m_pts->m_i_dims; j++)
			//	o << " " << m_pts->m_points[i][j];
			
			id = m_pid_to_cid[i];

			o << i << " " << id << endl;
		}

		int sum_points = 0;
		int noise = 0;
		for(i = 0; i < m_clusters.size(); i++)
		{
			sum_points += m_clusters[i].size();
			//cout << i << "(" << m_clusters[i].size() << ") ";
		}
	
		for (i = 0; i < m_pts->m_i_num_points; i++)
		{
			if(m_noise[i])
				noise++;
		}	
		
		cout << "Total points " << noise + sum_points << " pt_in_cls " << sum_points << " noise " << noise << endl;
		cout << "Number of clusters: " << m_clusters.size() << endl;
	}

	void ClusteringAlgo::writeClusters_uf(ostream& o)
	{
		// writing point id and cluster id pairs per line, noise has cluster id 0	
		vector <int> clusters;
		clusters.resize(m_pts->m_i_num_points, 0);

		int i, j, sum_points = 0, noise = 0, root, rootcount = 0, tmp;

		for(i = 0; i < m_pts->m_i_num_points; i++)
		{
			root = g_m_parents[i];

			// get the number of trees
			if(i == g_m_parents[i])
				rootcount++;

			// get the root of the tree containing i	
			while(root != g_m_parents[root])
				root = g_m_parents[root];

			// compress the tree to reduce the height of the branch of the tree to 1
			j = i;
			while(g_m_parents[j] != root)
			{
				tmp  = g_m_parents[j];
				g_m_parents[j] = root;
				j = tmp;
			}

			//m_parents[i] = root;
			// count the number of vertex in this tree
			clusters[root]++;
		}

		//cout << "clusters" << endl;
		int count = 0;
		for(i = 0; i < m_pts->m_i_num_points; i++)
		{
			if(clusters[i] == 1)
			{	// implies singleton
				// vertex i is a noise
				clusters[i] = 0;
				noise++;
			}
			else if(clusters[i] > 1)
			{
				// get the size of cluster count: this will happen only if i is a root
				count++;
				sum_points += clusters[i];
				clusters[i] = count;
			}
			// skip if i is not a root
		}

		// write point id and cluster ids to file
		for(i = 0; i < m_pts->m_i_num_points; i++)
		{
			//for (j = 0; j < m_pts->m_i_dims; j++)
            		//	o << " " << m_pts->m_points[i][j];	
		
			o << i << " " << clusters[g_m_parents[i]];
			if (g_m_corepoint[i])
				o << " core";
			if (clusters[g_m_parents[i]] == 0)
				o << " noise";
			o << endl;
		}

		cout << "Total points " << noise + sum_points << " pt_in_cls " << sum_points << " noise " << noise << endl;
		cout << "Number of clusters: " << count << endl;

		clusters.clear();
	}

	void run_dbscan_algo_uf(ClusteringAlgo& dbs, int threads)
	{			
		int tid;
 
        	// initialize some parameters
		dbs.m_clusters.clear();

		// get the neighbor of the first point and print them

		//cout << "DBSCAN ALGORITHMS============================" << endl;

		// assign parent to itestf
#if 0
		dbs.m_parents.resize(dbs.m_pts->m_i_num_points, -1);
		dbs.m_member.resize(dbs.m_pts->m_i_num_points, 0);
		dbs.m_corepoint.resize(dbs.m_pts->m_i_num_points, 0);
#else
		long* parentVecs = (long*)mw_malloc1dlong(dbs.m_pts->m_i_num_points);
		//for (int i = 0; i < dbs.m_pts->m_i_num_points; i++)
		//	parentVecs[i] = -1;
		//mw_replicated_init((long*)&g_m_parents, (long)parentVecs);
		//g_m_parents = (long*)malloc(sizeof(long) * dbs.m_pts->m_i_num_points);
		for (int i = 0; i < dbs.m_pts->m_i_num_points; i++)
			{
			  parentVecs[i] = -1;
			}
		mw_replicated_init((long*)&g_m_parents, (long)parentVecs);
		long* memberVecs = (long*)mw_malloc1dlong(dbs.m_pts->m_i_num_points);
		for (int i = 0; i < dbs.m_pts->m_i_num_points; i++)
			memberVecs[i] = 0;
		mw_replicated_init((long*)&g_m_member, (long)memberVecs);
		long* corepointVecs = (long*)mw_malloc1dlong(dbs.m_pts->m_i_num_points);
		for (int i = 0; i < dbs.m_pts->m_i_num_points; i++)
			corepointVecs[i] = 0;
		mw_replicated_init((long*)&g_m_corepoint, (long)corepointVecs);
#endif

#if 0
		int sch, maxthreads = omp_get_max_threads();
#else
		int sch, maxthreads;
		maxthreads = threads;
		cout << "Running with " << maxthreads << " parallel threads, including merge" << endl;
#endif
		
		if(dbs.m_pts->m_i_num_points % maxthreads == 0)
			sch = dbs.m_pts->m_i_num_points/maxthreads;
		else
			sch = dbs.m_pts->m_i_num_points/maxthreads + 1;
		
#if 0
		vector < vector <int > > merge; // 2D:  threads x points
		vector <int> init;
		merge.resize(maxthreads, init);
		for(tid = 0; tid < maxthreads; tid++)
			merge[tid].reserve(dbs.m_pts->m_i_num_points);
#else
		vector <int> init;
		//vector<int>* mergeVecs = (vector<int>*)mw_malloc2d(maxthreads,sizeof(vector<int>*));	// vectors are paired
		vector<int>** mergeVecs = (vector<int>**)mw_malloc1dlong(maxthreads);
		//LOG_DEBUG( "mergeVecs %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx\n", &mergeVecs[0], &mergeVecs[1], &mergeVecs[2], &mergeVecs[3], &mergeVecs[4], &mergeVecs[5], &mergeVecs[6], &mergeVecs[7], &mergeVecs[8] );
		//LOG_DEBUG( "mergeVecs[0] is %lx\n", mergeVecs[0] );
		//LOG_DEBUG( "mergeVecs[1] is %lx\n", mergeVecs[1] );
		//vector<int>* testVecs = (vector<int>*)malloc(sizeof(vector<int>*) * 10);
		//long* testVecs = (long*)mw_malloc1dlong(10);
		//LOG_DEBUG( "sizeof long* is %d, sizeof vec* is %d\n", sizeof(long*), sizeof(vector<int>*));
		//LOG_DEBUG( "testVecs %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx\n", &testVecs[0], &testVecs[1], &testVecs[2], &testVecs[3], &testVecs[4], &testVecs[5], &testVecs[6], &testVecs[7], &testVecs[8] );
		mw_replicated_init((long*)&merge, (long)mergeVecs);
		//merge = (vector<int>**)malloc(sizeof(vector<int>**) * maxthreads);
		//		merge = (vector<int>**)mw_mallocrepl(sizeof(vector<int>*) * maxthreads);
		//merge = (vector<int>**)mw_malloc1dlong(maxthreads);
		//LOG_DEBUG( "merge[0] is %lx\n", merge[0] );
		//LOG_DEBUG( "merge[1] is %lx\n", merge[1] );

		//cilk_spawn_at(&merge[0]) test_thread( dbs, merge, 0 );
		//cilk_sync;

		//LOG_DEBUG( "merge[0] is %lx\n", merge[0] );
		//LOG_DEBUG( "merge[1] is %lx\n", merge[1] );

		//LOG_DEBUG( "Debug exit\n" );
		//exit(0);

#endif

#if REPL_KDTREE
		vector<int>* ind = m_kdtree->getIndex();		
#else
		vector<int>* ind = dbs._m_kdtree->getIndex();
#endif

#if 0
		double start = omp_get_wtime();	
#else
		volatile uint64_t start = CLOCK();
#endif
		emu_dbscan(dbs, sch, ind, merge, maxthreads);

		// merge the trees that have not been merged yet
#if 0
		int v1, v2, size;
		double stop = omp_get_wtime() ;
		cout << "Local computation took " << stop - start << " seconds." << endl;
#else
		volatile uint64_t stop = CLOCK();
		cout << "Local computation took " << stop - start << " cycles." << endl;
#endif

		//allocate and initiate locks
#if 0
    		omp_lock_t *nlocks;
		nlocks = (omp_lock_t *) malloc(dbs.m_pts->m_i_num_points*sizeof(omp_lock_t));

		//start = stop;
		start = omp_get_wtime();

		#pragma omp parallel for private(i) shared(nlocks)
    		for(i = 0; i < dbs.m_pts->m_i_num_points; i++) 
      			omp_init_lock(&nlocks[i]); // initialize locks

		#pragma omp parallel for shared(maxthreads, merge, nlocks) private(i, v1, v2, root1, root2, size, tid)
#else
		start = CLOCK();
		//hooks_region_begin("merge");
#endif
		emu_merge(dbs, merge, maxthreads);
#if 0
		stop = omp_get_wtime();
		free(nlocks);
		cout << "Merging took " << stop - start << " seconds."<< endl;
#else
		stop = CLOCK();
		//double mergeTime = hooks_region_end();
		//cout << "Merging took " << stop - start << " cycles."<< endl;
		LOG_DEBUG( "There were %d distributed merges\n", distMerges );
		long distCycles = stop - start;
		LOG_DEBUG( "Distributed merging took %d cycles\n", distCycles );
		//cout << "Merging took " << mergeTime << " time."<< endl;
		if (distMerges > 0)
			LOG_DEBUG( "Average cycle count per merge was %d\n", distCycles / distMerges );
#endif

		for(tid = 0; tid < maxthreads; tid++)
		  merge[tid]->clear();

#if 0
		merge.clear();
#endif
	}
#if 0
	void test_thread(ClusteringAlgo& dbs, vector<int> **merge, int tid)
	{
		merge[tid] = new vector<int>;
		merge[tid]->reserve(dbs.m_pts->m_i_num_points);
		merge[tid]->push_back(1234);
	}
#endif

	void emu_merge(ClusteringAlgo& dbs, vector<int> **merge, int maxthreads)
	{
#if 8
		for(int tid = 0; tid < maxthreads; tid++)
		{
			cilk_spawn_at(merge[tid]) emu_merge_thread(dbs, merge, tid);
		}
#else
		cilk_spawn_at(merge[1]) emu_merge_thread(dbs, merge, 1);
		emu_merge_thread(dbs, merge, 0);
#endif
		cilk_sync;
	}
	void emu_merge_thread(ClusteringAlgo& dbs, vector<int> **merge, int tid)
	{
		int size = merge[tid]->size()/2;
		if (tid == 1)
			cout << "Distributed merges for thread 1:  " << size << endl;
		ATOMIC_ADDM( &distMerges, size );

		for(int i = 0; i < size; i++)
		{
			int v1 = (*merge[tid])[2 * i];
			int v2 = (*merge[tid])[2 * i + 1];
	
			int con = 0;
			if(g_m_corepoint[v2] == 1)
				con = 1;
			else if(g_m_member[v2] == 0)
			{
#if 0
				omp_set_lock(&nlocks[v2]);
				if(dbs.m_member[v2] == 0) // if v2 is not a member yet
				{
					con = 1;
					dbs.m_member[v2] = 1;
				}
				omp_unset_lock(&nlocks[v2]);
#else
				if (ATOMIC_SWAP(&g_m_member[v2],1) == 0)
					con = 1;
#endif
			}

			if(con == 1)
			{
			
				// lock based approach for merging
				long root1 = v1;
				long root2 = v2;

				// REMS algorithm with splicing compression techniques
				while (g_m_parents[root1] != g_m_parents[root2]) 
				{
					if (g_m_parents[root1] < g_m_parents[root2])
					{
						if(g_m_parents[root1] == root1) // root1 is a root
						{
#if 0
							omp_set_lock(&nlocks[root1]);
							int p_set = false;
							if(dbs.m_parents[root1] == root1) // if root1 is still a root
							{
								dbs.m_parents[root1] = dbs.m_parents[root2];
								p_set = true;
							}

							omp_unset_lock(&nlocks[root1]);
#else
							int p_set = false;
							if (ATOMIC_CAS((long*)(&g_m_parents[root1]),g_m_parents[root2],(long)root1) == root1)
								p_set = true;
#endif
							if (p_set) // merge successful
								break;
						}

						// splicing
						int z = g_m_parents[root1];
						g_m_parents[root1] = g_m_parents[root2];
						root1 = z;
						// root1 = dbs.m_parents[root1];
					}
					else
					{
						if(g_m_parents[root2] == root2) // root2 is a root
						{
							int p_set = false;
#if 0
							omp_set_lock(&nlocks[root2]);
							if(dbs.m_parents[root2] == root2) // check if root2 is a root
							{
								dbs.m_parents[root2] = dbs.m_parents[root1];
								p_set = true;
							}
							omp_unset_lock(&nlocks[root2]);
#else
							if (ATOMIC_CAS((long*)(&g_m_parents[root2]),g_m_parents[root1],(long)root2) == root2)
								p_set = true;
#endif
							if (p_set) // merge successful
								break;
						}
						
							//splicing
						int z = g_m_parents[root2];
						g_m_parents[root2] = g_m_parents[root1];
						if (root1 == 3213)
							cout << root2 << " now points to 3213" << endl;
						root2 = z;

						//root2 = dbs.m_parents[root2];
					}	
				}
			}
		}
	}	// emu_merge_thread()

	void emu_dbscan(ClusteringAlgo& dbs, int sch, vector<int>* ind, vector<int> **merge, int maxthreads)
	{
	#pragma omp parallel private(root, root1, root2, tid, ne, npid, i, j, pid) shared(sch, ind) //, prID)
		for (int tid = 0; tid < maxthreads; tid++)
			{
			cilk_spawn_at(&merge[tid]) emu_dbscan_thread(dbs,sch,ind,merge,tid);
			}
		cilk_sync;
	}
	
	void emu_dbscan_thread(ClusteringAlgo& dbs, int sch, vector<int>* ind, vector<int> **merge, int tid)
	{
		int pid, npid, i, j, root, root1, root2;
		int lower, upper;
		kdtree2_result_vector ne;
#if 0
		tid = omp_get_thread_num();
#endif

		
		vector < int > prID;
		
		merge[tid] = new vector<int>(); // create on this tid's starting nodelet

		//		printf("PRINT 1: THREAD %d IS ON NODELET %d \n", tid, NODE_ID());
		
		prID.resize(dbs.m_pts->m_i_num_points, -1);
		merge[tid]->reserve(dbs.m_pts->m_i_num_points);	// init merge vectors
		
		lower = sch * tid;
		upper = sch * (tid + 1);
		
		if(upper > dbs.m_pts->m_i_num_points)
			upper = dbs.m_pts->m_i_num_points;

		int t1count = 0;
		for(i = lower; i < upper; i++)
		{
			pid = (*ind)[i]; // CAN TRY RANDOMLY SELECTING
			g_m_parents[pid] = pid;
			prID[pid] = tid;
			if (tid == 1)
				t1count++;
		}

			//cout << "prID has " << t1count << " for t1" << endl;
		LOG_DEBUG( "%d prID has %d for t1\n", tid, t1count );

		#pragma omp barrier
		
		LOG_DEBUG( "%d About to start main dbscan thread loop\n" , tid);



		for(i = lower; i < upper; i++)
		  {

			pid = (*ind)[i];	// get the next point id to process

			ne.clear();
			// create the neighbor list for this point
#if REPL_KDTREE
			m_kdtree->r_nearest_around_point(pid, 0, dbs.m_epsSquare, ne);
#else
			dbs._m_kdtree->r_nearest_around_point(pid, 0, dbs.m_epsSquare, ne);
#endif
			
			if(ne.size() >= dbs.m_minPts)
			{	// this is a core point
				//if (pid == 3213)
					//LOG_DEBUG("3213 has %d neighbors\n", ne.size());
					//cout << "3213 has " << ne.size() << " neighbors" << endl;
				g_m_corepoint[pid] = 1;	// flag as core point
				g_m_member[pid] = 1;	// flag as cluster member
				
				// get the root containing pid
				root = pid;

				for (j = 0; j < ne.size(); j++)
				{	// go through points in the neighborhood
					npid= ne[j].idx;
					if(prID[npid] != tid)
					{	// distributed merge
					  //					        LOG_DEBUG("\tDistributed Merge!");
						merge[tid]->push_back(pid);
						merge[tid]->push_back(npid);
						continue;
					}

					// do a local merge
					// get the root containing npid
					root1 = npid;
					root2 = root;

					if(g_m_corepoint[npid] == 1 || g_m_member[npid] == 0)
					{
						g_m_member[npid] = 1;

						// REMS algorithm to merge the trees
						while(g_m_parents[root1] != g_m_parents[root2])
						{
							if(g_m_parents[root1] < g_m_parents[root2])
							{
								if(g_m_parents[root1] == root1)
								{	// root1 is its own parent:  merge root1 tree into root2 tree
									g_m_parents[root1] = g_m_parents[root2];
									if (root2 == 3213)
										cout << root1 << " now points to 3213" << endl;
									root = g_m_parents[root2];
									break;
								}

								// splicing
								// merge lower part of root1 tree into root2 tree
								int z = g_m_parents[root1];
								g_m_parents[root1] = g_m_parents[root2];
								root1 = z;
								//if (root2 == 3213)
								//	cout << root1 << " now points to 3213" << endl;
							}
							else
							{	// otherwise do the same but merge root2 into root1
								if(g_m_parents[root2] == root2)
								{
									g_m_parents[root2] = g_m_parents[root1];
									root = g_m_parents[root1];
									break;
								}

								// splicing
								int z = g_m_parents[root2];
								g_m_parents[root2] = g_m_parents[root1];
								root2 = z;
							}
						}
					}
				}
			}
		}
		ne.clear();
		LOG_DEBUG( "End of main dbscan thread loop for tid %d\n", tid );
	}	// emu_dbscan
   
	void run_dbscan_algo(ClusteringAlgo& dbs)
	{
		// classical DBSCAN algorithm (only sequential)
		int i, pid, j, k, npid;
		int cid = 1; // cluster id
		vector <int> c;
		c.reserve(dbs.m_pts->m_i_num_points);

           	// initialize some parameters
		dbs.m_noise.resize(dbs.m_pts->m_i_num_points, false);
            	dbs.m_visited.resize(dbs.m_pts->m_i_num_points, false);		
		dbs.m_pid_to_cid.resize(dbs.m_pts->m_i_num_points, 0);
		dbs.m_clusters.clear();

		// get the neighbor of the first point and print them

		//cout << "DBSCAN ALGORITHMS============================" << endl;

		kdtree2_result_vector ne;
		kdtree2_result_vector ne2;
		//kdtree2_result_vector ne3;
		ne.reserve(dbs.m_pts->m_i_num_points);
		ne2.reserve(dbs.m_pts->m_i_num_points);

		vector<int>* ind = dbs._m_kdtree->getIndex();

#if 0
		double start = omp_get_wtime() ;		
#endif

		for(i = 0; i < dbs.m_pts->m_i_num_points; i++)
		{
			pid = (*ind)[i];

			if (!dbs.m_visited[pid])
			{
				dbs.m_visited[pid] = true;
				ne.clear();
#if REPL_KDTREE
				m_kdtree->r_nearest_around_point(pid, 0, dbs.m_epsSquare, ne);
#else
				dbs._m_kdtree->r_nearest_around_point(pid, 0, dbs.m_epsSquare, ne);
#endif
				
				if(ne.size() < dbs.m_minPts)
					dbs.m_noise[pid] = true;
				else
				{
					// start a new cluster
					c.clear();
					c.push_back(pid);
					dbs.m_pid_to_cid[pid] = cid;

					// traverse the neighbors
					for (j = 0; j < ne.size(); j++)
					{
						npid= ne[j].idx;

						// not already visited
						if(!dbs.m_visited[npid])
						{
							dbs.m_visited[npid] = true;
	
							// go to neighbors
							ne2.clear();
#if REPL_KDTREE
							m_kdtree->r_nearest_around_point(npid, 0, dbs.m_epsSquare, ne2);
#else
							dbs._m_kdtree->r_nearest_around_point(npid, 0, dbs.m_epsSquare, ne2);
#endif
							

							// enough support
							if (ne2.size() >= dbs.m_minPts)
							{
								// join
								for(k = 0; k < ne2.size(); k++)
									ne.push_back(ne2[k]);
							}
						}

						// not already assigned to a cluster
						if (!dbs.m_pid_to_cid[npid])
						{
							c.push_back(npid);
							dbs.m_pid_to_cid[npid]=cid;
							dbs.m_noise[npid] = false;
						}
					}

					dbs.m_clusters.push_back(c);
					cid++;
				}
					
			}
		}
		
#if 0
	        double stop = omp_get_wtime();
        	cout << "Local computation took " << stop - start << " seconds." << endl;
#endif
		cout << "No merging stage in classical DBSCAN"<< endl;
		ind = NULL;
		ne.clear();
		ne2.clear();
	}
};

