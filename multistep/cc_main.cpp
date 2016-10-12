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


using namespace std;

#include <cstdlib>
#include <fstream>
#include <sys/time.h>
#include <string.h>
#include <vector>
#include <omp.h>

#define VERBOSE 0
#define DEBUG 0
#define VERIFY 0
#define TIMING 1

#define THREAD_QUEUE_SIZE 2048
#define ALPHA 15.0
#define BETA 25

double timer()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

typedef struct graph {
  int n;
  unsigned m;
  int* out_array;
  unsigned* out_degree_list;
} graph;
#define out_degree(g, n) (g.out_degree_list[n+1] - g.out_degree_list[n])
#define out_vertices(g, n) &g.out_array[g.out_degree_list[n]]

#include "cc_trim.cpp"
#include "cc_bfs.cpp"
#include "cc_color.cpp"
#include "cc_verify.cpp"

void read_edge(char* filename,
  int& n, unsigned& m,
  int*& srcs, int*& dsts)
{
  ifstream infile;
  string line;
  infile.open(filename);

  getline(infile, line, ' ');
  n = atoi(line.c_str());
  getline(infile, line);
  m = strtoul(line.c_str(), NULL, 10);

  int src, dst;
  unsigned counter = 0;

  m *= 2;
  srcs = new int[m];
  dsts = new int[m];
  for (unsigned i = 0; i < m/2; ++i)
  {
    getline(infile, line, ' ');
    src = atoi(line.c_str());
    getline(infile, line);
    dst = atoi(line.c_str());

    srcs[counter] = src;
    dsts[counter] = dst;
    ++counter;
    srcs[counter] = dst;
    dsts[counter] = src;
    ++counter;
  }

  infile.close();
}

void create_csr(int n, unsigned m, 
  int* srcs, int* dsts,
  int*& out_array, unsigned*& out_degree_list,
  int& max_deg_vert, double& avg_degree)
{
  out_array = new int[m];
  out_degree_list = new unsigned[n+1];

  for (unsigned i = 0; i < m; ++i)
    out_array[i] = 0;
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;

  unsigned* temp_counts = new unsigned[n];
  for (int i = 0; i < n; ++i)
    temp_counts[i] = 0;
  for (unsigned i = 0; i < m; ++i)
    ++temp_counts[srcs[i]];
  for (int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  copy(out_degree_list, out_degree_list + n, temp_counts);
  for (unsigned i = 0; i < m; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];
  delete [] temp_counts;

  avg_degree = 0.0;
  unsigned max_degree = 0;
  for (int i = 0; i < n; ++i)
  {
    unsigned degree = out_degree_list[i+1] - out_degree_list[i];
    avg_degree += (double)degree;
    if (degree > max_degree)
    {
      max_deg_vert = i;
      max_degree = degree;
    }
  }
  avg_degree /= (double)n;

#if DEBUG
  printf("max deg vert: %d, avg_degree %9.2lf\n", max_deg_vert, avg_degree);
#endif
}

void output_cc(graph& g, int* cc_maps, char* output_file)
{
  std::ofstream outfile;
  outfile.open(output_file);

  for (int i = 0; i < g.n; ++i)
    outfile << cc_maps[i] << std::endl;

  outfile.close();
}

void print_usage(char** argv)
{
  printf("Usage: %s [graph] [optional: output file]\n", argv[0]);
  exit(0);
}

void run_cc(graph& g, int*& cc_maps,
  int max_deg_vert, double avg_degree)
{

  int num_trim;
  int num_bfs;
  int num_color;

#if VERBOSE
  printf("Performing Trim step ... \n");
  double elt = timer();
  double start_time = elt;
#endif
  
  cc_maps = new int[g.n];
  bool* visited = new bool[g.n];
  int* valid_verts = new int[g.n];
  int num_valid;

  num_trim = cc_trim(g, visited, 
    valid_verts, num_valid, 
    cc_maps);

#if VERIFY
  cc_verify(g, cc_maps);  
#endif  

#if VERBOSE
  elt = timer() - elt;
  printf("\tDone, %9.6lf, %d verts trimmed\n", elt, num_trim);
  printf("Performing BFS step ... \n");
  elt = timer();
#endif

  num_bfs = cc_bfs(g, visited, 
    valid_verts, num_valid, 
    max_deg_vert, avg_degree,
    cc_maps);

#if VERIFY
  cc_verify(g, cc_maps);  
#endif  

#if VERBOSE
  elt = timer() - elt;
  printf("\tDone, %9.6lf, %d verts found\n", elt, num_bfs);
  printf("Performing Coloring step ... \n");
  elt = timer();
#endif

  num_color = cc_color(g, visited,
    valid_verts, num_valid,
    cc_maps);

#if VERBOSE
  elt = timer() - elt;
  printf("\tDone, %9.6lf, %d ccs found\n", elt, num_color);
  start_time = timer() - start_time;
  printf("TOTAL: %9.6lf\n", start_time);
#endif

}

int main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  if (argc < 2)
    print_usage(argv);

  int* srcs;
  int* dsts;
  int n;
  unsigned m;
  int* out_array;
  unsigned* out_degree_list;
  int* cc_maps;
  int max_deg_vert;
  double avg_degree;

#if VERBOSE
  double elt, start_time;
  printf("Reading %s ... ", argv[1]);
  elt = timer();
#endif

  read_edge(argv[1], n, m, srcs, dsts);

#if VERBOSE
  elt = timer() - elt;
  printf("Done, %9.6lf\n", elt);
  printf("Creating graph ... ");
  elt = timer();
#endif

  create_csr(n, m, srcs, dsts, 
    out_array, out_degree_list,
    max_deg_vert, avg_degree);
  graph g = {n, m, out_array, out_degree_list};
  delete [] srcs;
  delete [] dsts;

#if VERBOSE
  elt = timer() - elt;
  printf("Done, %9.6lf, n: %d, m: %u\n", elt, n, m/2);
  printf("Doing Multistep ... ");
  elt = timer();
#endif

#if TIMING
  double exec_time = timer();
#endif

  run_cc(g, cc_maps, 
    max_deg_vert, avg_degree);

#if TIMING
  exec_time = timer() - exec_time;
  printf("Multistep CC time: %9.6lf\n", exec_time);
#endif

#if VERBOSE
  elt = timer() - elt;
  printf("Done, %9.6lf\n", elt);
#endif

  cc_verify(g, cc_maps);

  if (argc == 3)
    output_cc(g, cc_maps, argv[2]);

  delete [] out_array;
  delete [] out_degree_list;
  delete [] cc_maps;

  return 0;
}
