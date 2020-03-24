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
#include "clusters.h"
#include "memoryweb.h"
#include "globals.h"

#if REPL_KDTREE
extern replicated long int NUM_REPLICATED_NODES;
replicated kdtree2 * m_kdtree;
#endif
//replicated kdtree2_node top_level_nodes[NUM_REPLICATED_NODES];

namespace NWUClustering
{
	Clusters::~Clusters()
	{
		if(m_pts)
		{
#if 0
			m_pts->m_points.clear();
			delete m_pts;
#else
			mw_free(m_pts->m_points);
			mw_free(m_pts);
			//free(m_pts->m_points);
			//free(m_pts);
#endif
			m_pts = NULL;
		}

#if REPL_KDTREE
		if(m_kdtree)
		{
		  mw_free(m_kdtree); //TODO: Make this free the actual trees/ local nodes, not just pointers
		  m_kdtree = NULL;
		  //mw_free(top_level_nodes);
		}
#endif
		if(_m_kdtree) {
		  delete _m_kdtree;
		  _m_kdtree = NULL;
		}
	}

	int Clusters::read_file(char* infilename, int isBinaryFile, int vectorCount)
	{
		ssize_t numBytesRead;
		int     i, j;
		int num_points, dims;

		if(isBinaryFile == 1)
        	{
			ifstream file (infilename, ios::in|ios::binary);
			if(file.is_open())
  			{
			  num_points = 0;
			  dims = 0;
				file.read((char*)&num_points, sizeof(int));
				file.read((char*)&dims, sizeof(int));
    				
				if ((vectorCount > 0) && (vectorCount < num_points))
					{
					LOG_DEBUG( "Setting num_points from %d to %d\n", num_points, vectorCount );
					num_points = vectorCount;
					}
				cout << "Points " << num_points << " dims " << dims << endl;

				// allocate memory for points
#if 0
				m_pts = new Points;				
				m_pts->m_i_dims = dims;
                		m_pts->m_i_num_points = num_points;
#else
				m_pts = (Points*)mw_mallocrepl(sizeof(Points));
				//m_pts = (Points*)malloc(sizeof(Points));
				mw_replicated_init((long*)&m_pts->m_i_dims,dims);
				mw_replicated_init((long*)&m_pts->m_i_num_points,num_points);
				//m_pts->m_i_dims = dims;
                		//m_pts->m_i_num_points = num_points;
#endif
				
				//allocate memory for the points
#if 0
                                m_pts->m_points.resize(num_points);
                                for(int ll = 0; ll < num_points; ll++)
                                        m_pts->m_points[ll].resize(dims);
#else
				double** l_m_points = (double**)mw_malloc2d(num_points,dims * sizeof(double));
				mw_replicated_init((long *)&m_pts->m_points, (long)l_m_points);
                                //m_pts->m_points = (double**)malloc(sizeof(double*) * num_points);
                                //for(int ll = 0; ll < num_points; ll++)
				//  m_pts->m_points[ll] = (double*)malloc(sizeof(double) * dims);
#endif
				
#if 0
				point_coord_type* pt;					
                        	pt = (point_coord_type*) malloc(dims * sizeof(point_coord_type));
#else
				float* pt;					
                        	pt = (float*) malloc(dims * sizeof(float));
#endif
                        
                        	for (i = 0; i < num_points; i++)
                        	{
#if 0
                                	file.read((char*)pt, dims*sizeof(point_coord_type));
#else
                                	file.read((char*)pt, dims*sizeof(float));
#endif
                                
                                	for (j = 0; j < dims; j++)
                                        	m_pts->m_points[i][j] = pt[j];
                        	}
			
				delete [] pt;	
				file.close();
				cout << "Completed file read" << endl;
  			}
			else
			{
				cout << "Error: no such file: " << infilename << endl;
				return -1;
			}
		}
		else
		{
			string line, line2, buf;
			ifstream file(infilename);
			stringstream ss;

			if (file.is_open())
  			{
				// get the first line and get the dimensions
				getline(file, line);
				line2 = line;
				ss.clear();				
				ss << line2;
			
				dims = 0;
				while(ss >> buf) // get the corordinate of the points
					dims++;

				// get point count
				num_points = 0;
				while (!file.eof())
				{
					if(line.length() == 0)
                                                continue;
					//cout << line << endl;
					num_points++;
					getline(file, line);
				}
				
				cout << "Points " << num_points << " dimensions " << dims << endl;
                               
				// allocate memory for points
#if 0
				m_pts = new Points;
				m_pts->m_points.resize(num_points);
                                for(int ll = 0; ll < num_points; ll++)
                                        m_pts->m_points[ll].resize(dims);
#else
				m_pts = (Points*)mw_mallocrepl(sizeof(Points));
				m_pts->m_points = (double**)mw_malloc2d(num_points,dims);
#endif
				

				file.clear();
				file.seekg (0, ios::beg);
				
				getline(file, line);

				i = 0;
    				while (!file.eof())
    				{
					if(line.length() == 0)
						continue;

					//cout << line << endl;
					ss.clear();
					ss << line;

					j = 0;
					while(ss >> buf && j < dims) // get the corordinate of the points
					{
						m_pts->m_points[i][j] = atof(buf.c_str());
						j++;
					}
					
					i++;
					getline(file, line);
    				}

    				file.close();
  			
#if 0
                                m_pts->m_i_dims = dims;
                                m_pts->m_i_num_points = num_points;
#else
				mw_replicated_init((long*)&m_pts->m_i_dims,dims);
				mw_replicated_init((long*)&m_pts->m_i_num_points,num_points);
#endif
			}                
			else
			{
                                cout << "Error: no such file: " << infilename << endl;
                                return -1;
			}			
		}
		
		return 0;		
	}


  
  /*std::string tabs(int n) {
    std::string ret = "";
    for(int i = 0; i < n; i++) {
      ret += "\t";
    }
    return ret;
    }*/

  /*
   * This function recursively makes local copies of the first NUM_REPLICATED_NODES kdtree nodes.
   * Left and right pointers will point to the local versions of top level nodes, to global otherwise
   * Cut values are replicated by shallow copy, intervals are not.
   */
#if REPL_KDTREE
  void create_local_copy(kdtree2_node * src, kdtree2_node * dst_array, long index, int depth=0) {
    if(index >= NUM_REPLICATED_NODES) {
      return;
    }
    //Create a local copy of the node
    //    cout << tabs(depth) << "Copying src from " << static_cast<const void *>(src) << " to " << static_cast<const void *>(&dst_array[index]) << endl;
    dst_array[index] = *src;
    //Update left pointer as needed and recurse
    if(2*index+1 < NUM_REPLICATED_NODES) {
      if(dst_array[index].left != NULL) {
	//	cout << tabs(depth) << "Pointing " << (const void *)&dst_array[index] << " left to " << (const void *)&dst_array[2*index+1] << endl;  
	dst_array[index].left = &dst_array[2*index+1];
	create_local_copy(src->left, dst_array, 2*index+1, depth+1);
      }
    }
    //Update right pointer as needed and recurse
    if(2*index+2 < NUM_REPLICATED_NODES) {
      if(dst_array[index].right != NULL) {
	//	cout << tabs(depth) << "Pointing " << (const void *)&dst_array[index] << " right to " << (const void *)&dst_array[2*index+2] << endl;  
	dst_array[index].right = &dst_array[2*index+2];
	create_local_copy(src->right, dst_array, 2*index+2, depth+1);
      }
    }
  }
#endif
		    

  int Clusters::build_kdtree(int bucketsize)
  {
		if(m_pts == NULL)
		{
			cout << "Point set is empty" << endl;
			return -1;
		}
		
		//Only compute kdtree once, keep around as the global copy
		_m_kdtree = new kdtree2(m_pts->m_points, m_pts->m_i_num_points, m_pts->m_i_dims, bucketsize, false);
		if(_m_kdtree == NULL)
		{
			cout << "Falied to allocate new kd tree" << endl;
			return -1;
		}		

		
#if REPL_KDTREE //==================================================================================
		
		
		//Allocate replicated kdtree
		m_kdtree = (kdtree2*)mw_mallocrepl(sizeof(kdtree2*));

		//Initialize the array seperately on each emu node
		//Pointers within each kdtree node need to point to the local versions children to avoid migrations
		LOG_DEBUG("=======================================\n");
		
		//TODO: parallelize?
		for(long node_index = 0; node_index < NODELETS(); node_index++) {

		  //Initialize local kdtree with pointer to local root
		  kdtree2 ** pointer_to_local_kdtree_memory = (kdtree2**)mw_get_nth(&m_kdtree, node_index);
		  //*pointer_to_local_kdtree_memory = new kdtree2(m_pts->m_points, m_pts->m_i_num_points, m_pts->m_i_dims, bucketsize, false);
		  *pointer_to_local_kdtree_memory = new kdtree2(_m_kdtree);
		  LOG_DEBUG("Created local version of kd_tree on node %ld at address %x\n", node_index, *pointer_to_local_kdtree_memory);
		  LOG_DEBUG("Address of its dim is %x\n", &((*pointer_to_local_kdtree_memory)->dim));
		  if(NUM_REPLICATED_NODES > 0) {
		  //Populate local array of kdtree nodes
		    kdtree2_node * local_nodes_buffer = (kdtree2_node*)malloc(NUM_REPLICATED_NODES * sizeof(kdtree2_node));
		    create_local_copy(_m_kdtree->root, local_nodes_buffer, 0);
		    ((kdtree2*)*pointer_to_local_kdtree_memory)->root = local_nodes_buffer; //only change root if we've actually replicated
		  }
		}

		LOG_DEBUG("=======================================\n");
#endif //==============================================================================================	     
		return 0;		
	} 
}
