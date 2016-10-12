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


class tarjan
{
public:
  tarjan(graph& g);
  tarjan(graph& g_in, bool* valid_in,
  int* valid_verts_in, int num_valid_in, 
  int* scc_maps_in);
  ~tarjan();

  int run();

  int run_valid();

  void scc_serial(int vert);

  void scc_serial_valid(int vert);

private:      
  tarjan() {}

  int local_scc_count;

  int* indices;
  int* lowlinks;
  int index;

  bool* in_stack;
  int* stack;
  int stack_pos;

  graph g;
  bool* valid;
  int* valid_verts;
  int num_valid;
  int* scc_maps;
  
  const static int NULL_VAL = -1;
};


tarjan::tarjan(graph& g)
{
  int num_verts = g.n;
  
  indices = new int[num_verts];
  lowlinks = new int[num_verts];
  in_stack = new bool[num_verts];
  stack = new int[num_verts];
  stack_pos = 0;
  for (int i = 0; i < num_verts; ++i)
    indices[i] = NULL_VAL;
  for (int i = 0; i < num_verts; ++i)
    lowlinks[i] = NULL_VAL;
  for (int i = 0; i < num_verts; ++i)
    in_stack[i] = false;

  index = 0;
  local_scc_count = 0;
}

tarjan::tarjan(graph& g_in, bool* valid_in,
  int* valid_verts_in, int num_valid_in, 
  int* scc_maps_in)
{
  g = g_in;
  valid = valid_in;
  valid_verts = valid_verts_in;
  num_valid = num_valid_in;
  scc_maps = scc_maps_in;

  int num_verts = g.n;  
  indices = new int[num_verts];
  lowlinks = new int[num_verts];
  in_stack = new bool[num_verts];
#pragma omp parallel for
  for (int i = 0; i < num_valid; ++i)
    indices[valid_verts[i]] = NULL_VAL;
#pragma omp parallel for
  for (int i = 0; i < num_valid; ++i)
    lowlinks[valid_verts[i]] = NULL_VAL;
#pragma omp parallel for
  for (int i = 0; i < num_valid; ++i)
    in_stack[valid_verts[i]] = false;

  stack = new int[num_verts];
  stack_pos = 0;

  index = 0;
  local_scc_count = 0;
}

tarjan::~tarjan()
{
  delete [] indices;
  delete [] lowlinks;
  delete [] in_stack;
  delete [] stack;
}

int tarjan::run()
{
  int num_verts = g.n;
  for (int v = 0; v < num_verts; ++v)
  {
    if (indices[v] == NULL_VAL)
    {
      scc_serial(v);
    }
  }

  return local_scc_count;
}

int tarjan::run_valid()
{
  for (int i = 0; i < num_valid; ++i)
  {
    int vert = valid_verts[i];
    if (indices[vert] == NULL_VAL)
    {
      scc_serial_valid(vert);
    }
  }

  return local_scc_count;
}


void tarjan::tarjan::scc_serial(int vert) 
{
  indices[vert] = index;
  lowlinks[vert] = index;
  ++index;

  stack[stack_pos++] = vert;
  in_stack[vert] = true;

  int out_degree = out_degree(g, vert);
  int* outs = out_vertices(g, vert);
  for(int i = 0; i < out_degree; ++i) 
  {
    int out = outs[i];
     
    if(indices[out] == NULL_VAL) 
    {
      scc_serial(out);
      lowlinks[vert] = lowlinks[vert] < lowlinks[out] ? lowlinks[vert] : lowlinks[out];
    }
    else if(in_stack[out]) 
    {
      lowlinks[vert] = lowlinks[vert] < indices[out] ? lowlinks[vert] : indices[out];
    }
  }

  if(lowlinks[vert] == indices[vert]) 
  {
    scc_maps[vert] = vert;
    int neighbor = NULL_VAL;       
    while(vert != neighbor) 
    {
      neighbor = stack[--stack_pos];
      scc_maps[neighbor] = vert;
      in_stack[neighbor] = false;
    }

    ++local_scc_count;
  }
}

void tarjan::scc_serial_valid(int vert) 
{
  indices[vert] = index;
  lowlinks[vert] = index;
  ++index;

  stack[stack_pos++] = vert;
  in_stack[vert] = true;

  int out_degree = out_degree(g, vert);
  int* outs = out_vertices(g, vert);
  for(int i = 0; i < out_degree; ++i) 
  {
    int out = outs[i];
    
    if (valid[out])
    { 
      if(indices[out] == NULL_VAL) 
      {
        scc_serial_valid(out);
        lowlinks[vert] = lowlinks[vert] < lowlinks[out] ? lowlinks[vert] : lowlinks[out];
      }
      else if(in_stack[out]) 
      {
        lowlinks[vert] = lowlinks[vert] < indices[out] ? lowlinks[vert] : indices[out];
      }
    }
  }

  if(lowlinks[vert] == indices[vert]) 
  {
    scc_maps[vert] = vert;
    int neighbor = NULL_VAL;       
    while(vert != neighbor) 
    {
      neighbor = stack[--stack_pos];
      scc_maps[neighbor] = vert;
      in_stack[neighbor] = false;
    }

    ++local_scc_count;
  }
}
