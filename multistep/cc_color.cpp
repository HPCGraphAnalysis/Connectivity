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


void cc_color_propagate(graph& g, bool* valid,
  int* valid_verts, int num_valid,
  int* cc_maps)
{
#if DEBUG
  double elt, elt2, start_time;
  start_time = timer();
#endif

  int num_verts = g.n;
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  copy(valid_verts, valid_verts + num_valid, queue);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_valid; ++i)
  { 
    int vert = valid_verts[i];
    cc_maps[vert] = vert;
    in_queue[vert] = true;
    in_queue_next[vert] = false;
  }

#if DEBUG
  elt = timer() - start_time;
  printf("init: %9.6lf\n", elt);
#endif

  int next_size = 0; 
  int queue_size = num_valid; 
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

#pragma omp for schedule(guided) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int vert = queue[i];
      in_queue[vert] = false;
      int color = cc_maps[vert];
      bool changed = false;

      int out_degree = out_degree(g, vert);
      int* outs = out_vertices(g, vert);
      for (int j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int out_color = cc_maps[out];

        if (valid[out] && color > out_color)
        {
          cc_maps[out] = color;
          changed = true;

          if (!in_queue_next[out])
          {
            in_queue_next[out] = true;
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
      
      if (changed && !in_queue_next[vert])
      {
        in_queue_next[vert] = true;
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
    queue_size = next_size;
    next_size = 0;
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp2 = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp2;
}
  } // end while
} // end parallel

  delete [] in_queue;
  delete [] in_queue_next;
  delete [] queue;
  delete [] queue_next;
}


int cc_color_count_roots(graph& g,
    int* valid_verts, int num_valid,
    int* cc_maps)
{
  int num_roots = 0;

#pragma omp parallel for schedule(static) reduction(+:num_roots)
  for (int i = 0; i < num_valid; ++i)
  {
    int vert = valid_verts[i];

    if (cc_maps[vert] == vert)
      ++num_roots;
  }

  return num_roots;  
}


int cc_color(graph& g, bool* visited,
    int* valid_verts, int& num_valid,
    int* cc_maps)
{
  int num_cc = 0;

  cc_color_propagate(g, visited,
    valid_verts, num_valid,
    cc_maps);

#if VERBOSE
  num_cc = cc_color_count_roots(g,
    valid_verts, num_valid,
    cc_maps);
#endif

  return num_cc;
}
