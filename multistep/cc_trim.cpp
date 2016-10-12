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


int cc_trim(graph& g, bool* valid, 
  int* valid_verts, int& num_valid, 
  int* cc_maps)
{
  int new_num_valid = 0;

#pragma omp parallel 
{ 
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

#pragma omp for
  for (int v = 0; v < g.n; ++v)
  {
    if (out_degree(g, v) < 1)
    {
      valid[v] = false;
      cc_maps[v] = v;
    }
    else
    {
      cc_maps[v] = -1;
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
