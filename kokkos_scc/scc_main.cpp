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

#include <Kokkos_Core.hpp>
//#if GPU
//#include <Kokkos_Cuda.hpp>
//#endif
#include <Kokkos_DualView.hpp>
//#include <Kokkos_Atomic.hpp>

#include <fstream>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU 0

#if GPU
  #define LOCAL_BUFFER_LENGTH 16
  #define WORK_CHUNK 2048
  #define WARP_SIZE 32
#else
  #define LOCAL_BUFFER_LENGTH 2048
  #define WORK_CHUNK 2048
#endif

#define CHUNK_SIZE 16

#define DELAYED_CUTOFF 16
#define DO_MAX 0

#define ALPHA 15.0
#define BETA 25

#define TIMING 0
#define VERBOSE 1
#define DEBUG 0
#define VERIFY 0

#define NUM_RUNS 5
#define TIMING_BFS 0
#define TIMING_COL 0
#define TIMING_COLBW 0
#define QUEUE_MULTIPLIER 1.25

#define MULTISTEP 0
#define GPU_ALG 0

char* graphname;
double runtime;
int alg_to_run;
int num_iters;

int root_start_out;
int root_end_out;
int root_start_in;
int root_end_in;

typedef Kokkos::DefaultExecutionSpace device_type;
typedef device_type::host_mirror_device_type host_type;
typedef Kokkos::TeamPolicy<device_type> team_policy;
typedef team_policy::member_type team_member;


struct join_max {
  typedef int value_type;

  KOKKOS_INLINE_FUNCTION
  void join( volatile value_type & update, 
    volatile const value_type & input) const
  { if (update < input) update = input; }
};

double timer()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}


#define out_degree(n) (out_degree_list[n+1] - out_degree_list[n])
#define in_degree(n) (in_degree_list[n+1] - in_degree_list[n])
#define out_vertices(n) &out_array[out_degree_list[n]]
#define in_vertices(n) &in_array[in_degree_list[n]]
#define out_vertice(n, j) out_array[out_degree_list[n]+j]
#define in_vertice(n, j) in_array[in_degree_list[n]+j]

/*
#include "scc_complete_trim.cpp"
*/
#include "scc_verify.cpp"
#include "color_baseline.cpp"
#include "color_manhattan_local.cpp"
#include "color_manhattan_global.cpp"
#include "scc_color.cpp"
#include "fwbw_baseline.cpp"
#include "fwbw_manhattan_local.cpp"
#include "fwbw_manhattan_global.cpp"
#include "scc_fwbw.cpp"
#include "scc_trim.cpp"
#include "scc_run.cpp"

void read_edge(char* filename,
  int& n, int& m,
  int*& srcs, int*& dsts)
{
  ifstream infile;
  string line;
  infile.open(filename);

  getline(infile, line, ' ');
  n = atoi(line.c_str());
  getline(infile, line);
  m = atoi(line.c_str());

  int src, dst;
  unsigned counter = 0;

  srcs = new int[m];
  dsts = new int[m];
  for (unsigned i = 0; i < m; ++i)
  {
    getline(infile, line, ' ');
    src = atoi(line.c_str());
    getline(infile, line);
    dst = atoi(line.c_str());

    srcs[counter] = src;
    dsts[counter] = dst;
    ++counter;
  }

  infile.close();
}

void create_csr(int n, int m, 
  int* srcs, int* dsts,
  int*& out_degree_list, int*& in_degree_list,
  int*& out_array, int*& in_array,
  int& max_deg_vert, double& avg_degree)
{
  out_degree_list = new int[n+1];
  in_degree_list = new int[n+1];
  out_array = new int[m];
  in_array = new int[m];

  for (int i = 0; i < m; ++i)
    out_array[i] = 0;
  for (int i = 0; i < m; ++i)
    in_array[i] = 0;  
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;
  for (int i = 0; i < n+1; ++i)
    in_degree_list[i] = 0;

  int* temp_counts = new int[n];
  for (int i = 0; i < n; ++i)
    temp_counts[i] = 0;
  for (int i = 0; i < m; ++i)
    ++temp_counts[srcs[i]];
  for (int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  for (int i = 0; i < n; ++i)
    temp_counts[i] = out_degree_list[i];
  for (int i = 0; i < m; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];

  for (int i = 0; i < n; ++i)
    temp_counts[i] = 0;
  for (int i = 0; i < m; ++i)
    ++temp_counts[dsts[i]];
  for (int i = 0; i < n; ++i)
    in_degree_list[i+1] = in_degree_list[i] + temp_counts[i];
  for (int i = 0; i < n; ++i)
    temp_counts[i] = in_degree_list[i];
  for (int i = 0; i < m; ++i)
    in_array[temp_counts[dsts[i]]++] = srcs[i];
  delete [] temp_counts;

  avg_degree = 0.0;
  double max_degree = 0.0;
  for (int i = 0; i < n; ++i)
  {
    int out_degree = out_degree_list[i+1] - out_degree_list[i];
    int in_degree = in_degree_list[i+1] - in_degree_list[i];
    double degree = (double)out_degree * (double)in_degree;
    avg_degree += (double)out_degree;
    if (degree > max_degree)
    {
      max_deg_vert = i;
      max_degree = degree;
    }
  }
  avg_degree /= (double)n;

  root_start_out = out_degree_list[max_deg_vert];
  root_end_out = out_degree_list[max_deg_vert+1];
  root_start_in = in_degree_list[max_deg_vert];
  root_end_in = in_degree_list[max_deg_vert+1];

#if DEBUG
  printf("max deg vert: %d (%d, %d), avg_degree %9.2lf\n", max_deg_vert, out_degree(max_deg_vert), in_degree(max_deg_vert), avg_degree);
#endif
}

void print_usage(char** argv)
{
  printf("Usage: %s [graph] [alg to run]\n", argv[0]);
  exit(0);
}



int main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  if (argc < 3)
    print_usage(argv);

  typedef Kokkos::View<bool*, device_type> bool_array;
  typedef Kokkos::View<int*, device_type> int_array;
  typedef Kokkos::View<unsigned*, device_type> unsigned_array;
  typedef Kokkos::View<const int*, Kokkos::MemoryRandomAccess> int_array_rand;
  typedef Kokkos::View<const unsigned*, Kokkos::MemoryRandomAccess> unsigned_array_rand;
  typedef Kokkos::View<int, device_type> int_type;
  typedef Kokkos::View<double, device_type> double_type;


  Kokkos::initialize();
/*
#if GPU
  host_type::initialize(1);
#endif
  device_type::initialize();
*/

  int n;
  int m;
  int* srcs;
  int* dsts;
  int* out_degree_list;
  int* in_degree_list;
  int* out_array;
  int* in_array;
  int max_deg_vert;
  double avg_degree;

#if VERBOSE
  double elt;
  printf("Reading %s ... ", argv[1]);
  elt = timer();
#endif

  read_edge(argv[1], n, m, srcs, dsts);
  alg_to_run = atoi(argv[2]);

#if VERBOSE
  elt = timer() - elt;
  printf("Done, %9.6lf\n", elt);
  printf("Creating graph ... ");
  elt = timer();
#endif

  create_csr(n, m, srcs, dsts, 
    out_degree_list, in_degree_list,
    out_array, in_array,
    max_deg_vert, avg_degree);
  delete [] srcs;
  delete [] dsts;


  int_type n_dev("n");
  int_array in_degree_list_dev("in degree list", n+1);
  int_array in_array_dev("in array", m);
  int_array out_degree_list_dev("out degree list", n+1);
  int_array out_array_dev("out array", m);
  int_array valid_verts_dev("valid verts", n*QUEUE_MULTIPLIER);
  bool_array valid_dev("valid", n);
  int_type num_valid_dev("num valid");
  int_type max_deg_vert_dev("max deg vert");
  double_type avg_degree_dev("avg degree");
  int_array scc_maps("scc maps", n);

  int_type::HostMirror n_host = Kokkos::create_mirror_view(n_dev);
   int_array::HostMirror in_degree_list_host = Kokkos::create_mirror_view(in_degree_list_dev);
  int_array::HostMirror in_array_host = Kokkos::create_mirror_view(in_array_dev);
  int_array::HostMirror out_degree_list_host = Kokkos::create_mirror_view(out_degree_list_dev);
  int_array::HostMirror out_array_host = Kokkos::create_mirror_view(out_array_dev); 
  int_array::HostMirror valid_verts_host = Kokkos::create_mirror_view(valid_verts_dev);
  bool_array::HostMirror valid_host = Kokkos::create_mirror_view(valid_dev);
  int_type::HostMirror num_valid_host = Kokkos::create_mirror_view(num_valid_dev);
  int_type::HostMirror max_deg_vert_host = Kokkos::create_mirror_view(max_deg_vert_dev);
  double_type::HostMirror avg_degree_host = Kokkos::create_mirror_view(avg_degree_dev);

  *n_host = n;
  for (int i = 0; i < n+1; ++i)
  {
    out_degree_list_host[i] = out_degree_list[i];
    in_degree_list_host[i] = in_degree_list[i];
  }
  for (int i = 0; i < m; ++i)
  {
    out_array_host[i] = out_array[i];
    in_array_host[i] = in_array[i]; 
  }
  *max_deg_vert_host = max_deg_vert;
  *avg_degree_host = avg_degree;


  Kokkos::deep_copy(n_dev, n_host);
  Kokkos::deep_copy(out_degree_list_dev, out_degree_list_host);
  Kokkos::deep_copy(out_array_dev, out_array_host);
  Kokkos::deep_copy(in_degree_list_dev, in_degree_list_host);
  Kokkos::deep_copy(in_array_dev, in_array_host);
  Kokkos::deep_copy(max_deg_vert_dev, max_deg_vert_host);
  Kokkos::deep_copy(avg_degree_dev, avg_degree_host);

#if VERBOSE
  elt = timer() - elt;
  printf("Done, %9.6lf, n: %d, m: %u\n", elt, n, m);
  printf("Doing Multistep ... ");
  elt = timer();
#endif

  double this_runtime = timer();

#if TIMING
  num_iters = atoi(argv[2]);
  graphname = argv[3];
#endif

  do_run<device_type>(n_dev,
    out_degree_list_dev, out_array_dev, 
    in_degree_list_dev, in_array_dev,
    max_deg_vert_dev, avg_degree_dev,
    scc_maps,
    valid_verts_dev, valid_dev, num_valid_dev);

  this_runtime = timer() - this_runtime;

#if VERBOSE
  elt = timer() - elt;
  printf("Done, %9.6lf\n", elt);
#endif



/*
  //device_type::finalize();
#if GPU
  host_type::finalize();
#endif
*/

  Kokkos::finalize();
  return 0;
}
