/*
//@HEADER
// *****************************************************************************
//
//       Multistep: (Strongly) Connected Components Algorithms
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/


bool* cc_bfs_search(graph& g, bool* valid, 
  int* valid_verts, int num_valid,
  int root, double avg_degree)
{
  int num_verts = g.n;
  int* queue = new int[num_valid];
  int* queue_next = new int[num_valid];
  int queue_size = 0; 
  int next_size = 0;
  int num_ccs = 0;
  bool use_hybrid = false;
  bool done_switch = false;

  bool* cc = new bool[num_verts];
#pragma omp parallel for
  for (int i = 0; i < num_valid; ++i)
    cc[valid_verts[i]] = false;

#if DEBUG
  double elt;
  int level = 0;
#endif

  queue[queue_size++] = root;
  cc[root] = true;
#pragma omp parallel
{
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

  while (queue_size)
  {
#if DEBUG
    elt = timer();
#endif

    if (!use_hybrid)
    {
  #pragma omp for schedule(guided) nowait
      for (int i = 0; i < queue_size; ++i)
      {
        int vert = queue[i];

        int out_degree = out_degree(g, vert);
        int* outs = out_vertices(g, vert);
        for (int j = 0; j < out_degree; ++j)
        {     
          int out = outs[j];  
          if (!cc[out] && valid[out])
          {
            cc[out] = true;
            thread_queue[thread_queue_size++] = out;

            if (thread_queue_size == THREAD_QUEUE_SIZE)
            {
#pragma omp atomic capture
              thread_start = next_size += thread_queue_size;
              
              thread_start -= thread_queue_size;
              for (int l = 0; l < thread_queue_size; ++l)
                queue_next[thread_start+l] = thread_queue[l];
              thread_queue_size = 0;
            }
          }
        }
      }
    }
    else
    {
  #pragma omp for schedule(guided) nowait
      for (int i = 0; i < num_valid; ++i)
      {
        int vert = valid_verts[i];
        if (!cc[vert] && valid[vert])
        {
          int out_degree = out_degree(g, vert);
          int* outs = out_vertices(g, vert);
          for (int j = 0; j < out_degree; ++j)
          {
            int out = outs[j];
            if (cc[out] && valid[out])
            {
              cc[vert] = true;
              thread_queue[thread_queue_size++] = vert;

              if (thread_queue_size == THREAD_QUEUE_SIZE)
              {
#pragma omp atomic capture
                thread_start = next_size += thread_queue_size;
                
                thread_start -= thread_queue_size;
                for (int l = 0; l < thread_queue_size; ++l)
                  queue_next[thread_start+l] = thread_queue[l];
                thread_queue_size = 0;
              }
              break;
            }
          }
        }
      }
    }

#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;
#pragma omp barrier

#pragma omp single
{   
    num_ccs += next_size;

#if DEBUG
    printf("num_ccs: %d   cur: %d\n", num_ccs, next_size);
#endif

    if (!done_switch && !use_hybrid)
    { 
      double edges_frontier = (double)next_size * avg_degree;
      double edges_remainder = (double)(num_verts - num_ccs) * avg_degree;

#if DEBUG
      printf("edge_front: %f  edge_rem: %f\n", edges_frontier, edges_remainder);
#endif

      if ((edges_remainder / ALPHA) < edges_frontier)
      {
#if DEBUG
        printf("\n=======switching to hybrid\n\n");
#endif
        use_hybrid = true;
      }
    }
    else
    {
      if (!done_switch && (num_verts / BETA) > next_size)
      {
#if DEBUG
        printf("\n=======switching back\n\n");
#endif
        use_hybrid = false;
        done_switch = true;
      }
    }

    queue_size = next_size;
    next_size = 0;
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
 
#if DEBUG
    ++level;
    elt = timer() - elt;
    printf("level: %d  num: %d  time: %9.6lf\n", level, queue_size, elt);
#endif
}
  }
}
  delete [] queue;
  delete [] queue_next; 

  return cc;
}


int update_valid(graph& g, bool* valid, bool* cc, 
  int*& valid_verts, int& num_valid,
  int* cc_maps, int root)
{  
  int new_num_valid = 0;
  int* new_valid_verts = new int[num_valid];

#pragma omp parallel 
{ 
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

#pragma omp for nowait
  for (int i = 0; i < g.n; ++i)
  {
    int vert = valid_verts[i];
    if (cc[vert])
    {
      valid[vert] = false;
      cc_maps[vert] = root;
    }
    else if (valid[vert])
    {
      thread_queue[thread_queue_size++] = vert;

      if (thread_queue_size == THREAD_QUEUE_SIZE)
      {
#pragma omp atomic capture
        thread_start = new_num_valid += thread_queue_size;
        
        thread_start -= thread_queue_size;
        for (int l = 0; l < thread_queue_size; ++l)
          new_valid_verts[thread_start+l] = thread_queue[l];
        thread_queue_size = 0;
      }
    }
  }

#pragma omp atomic capture
  thread_start = new_num_valid += thread_queue_size;
  
  thread_start -= thread_queue_size;
  for (int l = 0; l < thread_queue_size; ++l)
    new_valid_verts[thread_start+l] = thread_queue[l];
}
  
  int num_cc = num_valid - new_num_valid;
  num_valid = new_num_valid;

  delete [] valid_verts;
  valid_verts = new_valid_verts;

  return num_cc;
}

int cc_bfs(graph& g, bool* valid, 
    int*& valid_verts, int& num_valid, 
    int max_deg_vert, double avg_degree,
    int* cc_maps)
{
  int num_cc = 0;

  bool* cc = cc_bfs_search(g, valid, 
    valid_verts, num_valid, 
    max_deg_vert, avg_degree);

  num_cc = update_valid(g, valid, cc, 
    valid_verts, num_valid,
    cc_maps, max_deg_vert);

  delete [] cc;

  return num_cc;
}
