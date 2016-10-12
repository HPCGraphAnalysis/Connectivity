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


int scc_trim_none(graph& g, bool* valid,
  int* valid_verts, int& num_valid,
  int* scc_maps)
{
#pragma omp parallel
{
#pragma omp for nowait
  for (int i = 0; i < g.n; ++i) valid[i] = true;
#pragma omp for nowait
  for (int i = 0; i < g.n; ++i) scc_maps[i] = -1;
}

  num_valid = g.n;

  return 0;
}

int scc_trim_simple(graph& g, bool* valid,
  int* valid_verts, int& num_valid,
  int* scc_maps)
{
  int trim_count = 0;

#pragma omp parallel for reduction(+:trim_count)
  for (int v = 0; v < g.n; ++v)
  {
    if (out_degree(g, v) < 1 || in_degree(g, v) < 1)
    {
      valid[v] = false;
      scc_maps[v] = v;
      ++trim_count;
    }
    else
    {
      valid[v] = true;
      scc_maps[v] = -1;
    }
  }

  return trim_count;
}

int scc_trim(graph& g, bool* valid, 
  int* valid_verts, int& num_valid, 
  int* scc_maps)
{
  int new_num_valid = 0;

#pragma omp parallel 
{ 
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

#pragma omp for schedule(guided) nowait
  for (int v = 0; v < g.n; ++v)
  {
    if (out_degree(g, v) < 1 || in_degree(g, v) < 1)
    {
      valid[v] = false;
      scc_maps[v] = v;
    }
    else
    {
      scc_maps[v] = -1;
      valid[v] = true;
      thread_queue[thread_queue_size++] = v;    

      if (thread_queue_size == THREAD_QUEUE_SIZE)
      {
#pragma omp atomic capture
        thread_start = new_num_valid += thread_queue_size;
        
        thread_start -= thread_queue_size;
        for (int l = 0; l < thread_queue_size; ++l)
          valid_verts[thread_start+l] = thread_queue[l];
        thread_queue_size = 0;
      }
    }
  }

#pragma omp atomic capture
  thread_start = new_num_valid += thread_queue_size;
  
  thread_start -= thread_queue_size;
  for (int l = 0; l < thread_queue_size; ++l)
    valid_verts[thread_start+l] = thread_queue[l];
}

  num_valid = new_num_valid;

  return (g.n - num_valid);
}

int scc_trim_complete(graph& g, bool* valid, 
  int*& valid_verts, int& num_valid,
  int* scc_maps)
{
  int num_verts = g.n;
  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size = num_verts;
  int next_size = 0;

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
  {
    valid[i] = true;
    queue[i] = i;
    in_queue_next[i] = false;
  }

  int trim_count = 0;
#pragma omp parallel
{
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

  while (queue_size)
  {
    thread_queue_size = 0;

#pragma omp for schedule(guided) reduction(+:trim_count)
    for (int i = 0; i < queue_size; ++i)
    {
      int vert = queue[i];
      in_queue[vert] = false;

      int out_degree = out_degree(g, vert);
      int in_degree = in_degree(g, vert);
      int* outs = out_vertices(g, vert);
      int* ins = in_vertices(g, vert);
      int out, in;
      bool has_out = false;
      bool has_in = false;

      for (int j = 0; j < out_degree; ++j)
      {
        out = outs[j];
        if (valid[out] && out != vert)
        {
          has_out = true;
          break;
        }
      }
      for (int j = 0; j < in_degree; ++j)
      {
        in = ins[j];
        if (valid[in] && in != vert)
        {
          has_in = true;
          break;
        }
      }

      if (!has_out && !has_in)
      {
        ++trim_count;
        scc_maps[vert] = vert;
        valid[vert] = false;
      }
      else if (!has_out)
      {
        ++trim_count;
        scc_maps[vert] = vert;
        valid[vert] = false;
        for (int j = 0; j < in_degree; ++j)
        {
          in = ins[j];
          if (valid[in] && !in_queue_next[in])
          {
            in_queue_next[in] = true;
            thread_queue[thread_queue_size++] = in;

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
      else if (!has_in)
      {
        ++trim_count;
        scc_maps[vert] = vert;
        valid[vert] = false;
        for (int j = 0; j < out_degree; ++j)
        {
          out = outs[j];

          if (valid[out] && !in_queue_next[out])
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
      else
        scc_maps[vert] = -1;
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
  }
}

  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;

  return trim_count;
}

