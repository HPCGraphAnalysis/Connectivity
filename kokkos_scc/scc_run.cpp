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



template< class ExecSpace >
void do_run(Kokkos::View<int, ExecSpace> n,
  Kokkos::View<int*, ExecSpace> out_degree_list,
  Kokkos::View<int*, ExecSpace> out_array,
  Kokkos::View<int*, ExecSpace> in_degree_list,
  Kokkos::View<int*, ExecSpace> in_array,
  Kokkos::View<int, ExecSpace> max_degree_vert,
  Kokkos::View<double, ExecSpace> avg_degree,
  Kokkos::View<int*, ExecSpace> scc_maps,
  Kokkos::View<int*, ExecSpace> valid_verts,
  Kokkos::View<bool*, ExecSpace> valid,
  Kokkos::View<int, ExecSpace> num_valid)
{
#if TIMING
for (int a = 0; a < 3; ++a)
{
char* algname = new char[100];
alg_to_run = a;
double total_time = 0.0;
for (int t = 0; t < num_iters; ++t)
{  
#endif 
  typename Kokkos::View<int, ExecSpace>::HostMirror host_n = Kokkos::create_mirror(n);
  typename Kokkos::View<int, ExecSpace>::HostMirror host_num_valid = Kokkos::create_mirror(num_valid);
  Kokkos::deep_copy(host_n, n);
  Kokkos::View<int*, ExecSpace> queue("queue", host_n()*QUEUE_MULTIPLIER);
  Kokkos::View<int*, ExecSpace> queue_next("queue next", host_n()*QUEUE_MULTIPLIER);
  Kokkos::View<int*, ExecSpace> offsets("offsets", host_n()*QUEUE_MULTIPLIER);
  Kokkos::View<int*, ExecSpace> offsets_next("offsets next", host_n()*QUEUE_MULTIPLIER);
  Kokkos::View<bool*, ExecSpace> in_queue("queue next", host_n());
  Kokkos::View<bool*, ExecSpace> in_queue_next("queue next", host_n());
  Kokkos::View<int*, ExecSpace> owner("owner", host_n());
  Kokkos::View<int*, ExecSpace> colors("colors", host_n());


#if VERBOSE
  printf("Performing Trim step ... \n");
  double elt = timer();
  double start_time = elt;
#endif

#if VERIFY 
  typename Kokkos::View<int*, ExecSpace>::HostMirror scc_host = Kokkos::create_mirror(scc_maps);
  typename Kokkos::View<int, ExecSpace>::HostMirror n_host = Kokkos::create_mirror(n);
  Kokkos::deep_copy(n_host, n);
  int* maps = new int[*n_host];
#endif

#if TIMING
  double this_time = timer();
#endif

  simple_trim<ExecSpace>(n,
    out_degree_list, in_degree_list,
    num_valid, valid_verts, valid,
    scc_maps);

#if VERBOSE
  elt = timer() - elt;
  Kokkos::deep_copy(host_num_valid, num_valid);
  printf("\tDone, %9.6lf, valid: %d\n", elt, host_num_valid());
  printf("Performing FWBW step ... \n");
  elt = timer();
#endif

#if VERIFY
  Kokkos::deep_copy(scc_host, scc_maps);
  for (int i = 0; i < n_host(); ++i)
    maps[i] = scc_host[i];
  scc_verify(*n_host, maps);
#endif

  do_fwbw<ExecSpace>(n,
    out_degree_list, out_array, 
    in_degree_list, in_array, 
    num_valid, valid_verts, valid,
    max_degree_vert, avg_degree,
    scc_maps,
    queue, queue_next, 
    offsets, offsets_next,
    in_queue, in_queue_next);

#if VERBOSE
  elt = timer() - elt;
  Kokkos::deep_copy(host_num_valid, num_valid);
  printf("\tDone, %9.6lf valid: %d\n", elt, host_num_valid());
  printf("Performing Coloring step ... \n");
  elt = timer();
#endif

#if VERIFY
  Kokkos::deep_copy(scc_host, scc_maps);
  for (int i = 0; i < n_host(); ++i)
    maps[i] = scc_host[i];
  scc_verify(*n_host, maps);
#endif

  do_coloring<ExecSpace>(n,
    out_degree_list, out_array,
    in_degree_list, in_array,
    num_valid, valid_verts, valid,
    scc_maps,
    queue, queue_next,
    offsets, offsets_next,
    in_queue, in_queue_next,
    colors, owner);

#if TIMING
  this_time = timer() - this_time;
#endif

#if VERBOSE
  elt = timer() - elt;
  Kokkos::deep_copy(host_num_valid, num_valid);
  printf("\tDone, %9.6lf, valid: %d\n", elt, host_num_valid());
  start_time = timer() - start_time;
  printf("TOTAL: %9.6lf\n", start_time);
#endif

#if VERIFY
  Kokkos::deep_copy(scc_host, scc_maps);
  for (int i = 0; i < n_host(); ++i)
    maps[i] = scc_host[i];
  scc_verify(*n_host, maps);
#endif
#if TIMING
  total_time += this_time;
} // end iter
double avg_time = total_time / (double)num_iters;
if (alg_to_run == 0) 
  printf("%s, Baseline, %2.4lf\n", graphname, avg_time);
else if (alg_to_run == 1) 
  printf("%s, ManhattanLocal, %2.4lf\n", graphname, avg_time);
else if (alg_to_run == 2) 
  printf("%s, ManhattanGlobal, %2.4lf\n", graphname, avg_time);

} // end alg
#endif
}
