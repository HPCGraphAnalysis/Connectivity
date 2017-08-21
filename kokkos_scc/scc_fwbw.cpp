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


/*
'####:'##::: ##:'####:'########:
. ##:: ###:: ##:. ##::... ##..::
: ##:: ####: ##:: ##::::: ##::::
: ##:: ## ## ##:: ##::::: ##::::
: ##:: ##. ####:: ##::::: ##::::
: ##:: ##:. ###:: ##::::: ##::::
'####: ##::. ##:'####:::: ##::::
....::..::::..::....:::::..:::::
*/
template< class ExecSpace >
struct fwbw_init {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_type root;
  int_type n;
  bool_array fw;
  bool_array scc;

  fwbw_init(int_type n_in,
    bool_array fw_in, bool_array scc_in,
    int_type root_in) 
  : n(n_in)
  , fw(fw_in), scc(scc_in)
  , root(root_in)
  {    
    typename int_type::HostMirror n_host = Kokkos::create_mirror(n);
    Kokkos::deep_copy(n_host, n);

    int num_teams = (n_host() + WORK_CHUNK - 1 ) / WORK_CHUNK;
    int team_size = team_policy::team_size_recommended(*this); //best guess
    team_policy policy(num_teams, team_size);
    Kokkos::parallel_for(policy , *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev  ) const
  {
    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = n() < end ? n() : end;
    int team_size = dev.team_size();

    for (int v = begin; v < end; v += team_size)
    {
      if (v == root())
      {
        fw[v] = true;
        scc[v] = true;
      }
      else
      {
        fw[v] = false;
        scc[v] = false;
      }
    }
  }
};

template< class ExecSpace >
struct fwbw_init_offsets {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_type root;
  int_type offsets_max_out;
  int_type offsets_max_in;

  int_type n;
  int_array out_degree_list;
  int_array in_degree_list;
  bool_array fw;
  bool_array scc;

  fwbw_init_offsets(int_type n_in,
    int_array out_degree_list_in, int_array in_degree_list_in,
    bool_array fw_in, bool_array scc_in,
    int_type root_in,
    int_type offsets_max_out_in, int_type offsets_max_in_in) 
  : n(n_in)
  , out_degree_list(out_degree_list_in), in_degree_list(in_degree_list_in)
  , fw(fw_in), scc(scc_in)
  , root(root_in)
  , offsets_max_out(offsets_max_out_in), offsets_max_in(offsets_max_in_in)
  {    
    typename int_type::HostMirror n_host = Kokkos::create_mirror(n);
    Kokkos::deep_copy(n_host, n);

    int num_teams = (n_host() + WORK_CHUNK - 1 ) / WORK_CHUNK;
    int team_size = team_policy::team_size_recommended(*this); //I have no idea if this is correct
    team_policy policy(num_teams, team_size);
    Kokkos::parallel_for(policy , *this);
  }


  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev  ) const
  {
    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = n() < end ? n() : end;
    int team_size = dev.team_size();

    for (int v = begin; v < end; v += team_size)
    {
      if (v == root())
      {
        fw[v] = true;
        scc[v] = true;
        offsets_max_out() = out_degree(v);
        offsets_max_in() = in_degree(v);
      }
      else
      {
        fw[v] = false;
        scc[v] = false;
      }
    }
  }
};



/*
'##::::'##:'########::'########:::::'###::::'########:'########:
 ##:::: ##: ##.... ##: ##.... ##:::'## ##:::... ##..:: ##.....::
 ##:::: ##: ##:::: ##: ##:::: ##::'##:. ##::::: ##:::: ##:::::::
 ##:::: ##: ########:: ##:::: ##:'##:::. ##:::: ##:::: ######:::
 ##:::: ##: ##.....::: ##:::: ##: #########:::: ##:::: ##...::::
 ##:::: ##: ##:::::::: ##:::: ##: ##.... ##:::: ##:::: ##:::::::
. #######:: ##:::::::: ########:: ##:::: ##:::: ##:::: ########:
:.......:::..:::::::::........:::..:::::..:::::..:::::........::
*/
template< class ExecSpace >
struct fwbw_update_valid {

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef ExecSpace device_type;

  int_type root;

  int_type n;
  int_array valid_verts;
  bool_array valid;
  int_type num_valid;
  int_type cur_valid;
  int_type test;

  bool_array scc;
  int_array scc_maps;

  fwbw_update_valid(int_type n_in,
    bool_array scc_in,
    int_type num_valid_in, int_array valid_verts_in,
    bool_array valid_in,
    int_type root_in,
    int_array scc_maps_in) 
  : n(n_in)
  , scc(scc_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in)
  , valid(valid_in)
  , root(root_in)
  , scc_maps(scc_maps_in) 
  , cur_valid("cur valid"), test("test")
  {    
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    typename int_type::HostMirror host_n = create_mirror(n);
    deep_copy(host_num_valid, num_valid);
    deep_copy(host_n, n);
    deep_copy(cur_valid, host_num_valid);

    int num_teams = (host_n() + WORK_CHUNK - 1 ) / WORK_CHUNK;
    int team_size = team_policy::team_size_recommended(*this);
    team_policy policy(num_teams, team_size);

    host_num_valid() = 0;
    deep_copy(num_valid, host_num_valid);

    Kokkos::parallel_for(policy , *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev  ) const
  {
    int local_buffer[ LOCAL_BUFFER_LENGTH ];
    int local_count = 0;
    int k = 0;

    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = n() < end ? n() : end;
    int team_size = dev.team_size();

    for (int i = begin; i < end; i += team_size)
    {  
      int vert = i;//valid_verts[i];
      if (scc[vert])
      {
        valid[vert] = false;
        scc_maps[vert] = root();
      }
      else if (valid[vert])
      {
        local_buffer[local_count++] = vert;
      }
      ++k;

      if (k == LOCAL_BUFFER_LENGTH)
      {
        const int team_offset = dev.team_scan(local_count, &num_valid());
        for (int l = 0; l < local_count; ++l)
          valid_verts[team_offset+l] = local_buffer[l];

        k = 0;
        local_count = 0;
      }
    }

    const int team_offset = dev.team_scan(local_count, &num_valid());
    for (int l = 0; l < local_count; ++l)
      valid_verts[team_offset+l] = local_buffer[l];
  }
};


/*
'########::'##::::'##:'##::: ##:
 ##.... ##: ##:::: ##: ###:: ##:
 ##:::: ##: ##:::: ##: ####: ##:
 ########:: ##:::: ##: ## ## ##:
 ##.. ##::: ##:::: ##: ##. ####:
 ##::. ##:: ##:::: ##: ##:. ###:
 ##:::. ##:. #######:: ##::. ##:
..:::::..:::.......:::..::::..::
*/
template < class ExecSpace >
void do_fwbw(
  Kokkos::View<int, ExecSpace> n,
  Kokkos::View<int*, ExecSpace> out_degree_list,
  Kokkos::View<int*, ExecSpace> out_array,
  Kokkos::View<int*, ExecSpace> in_degree_list,
  Kokkos::View<int*, ExecSpace> in_array,
  Kokkos::View<int, ExecSpace> num_valid,
  Kokkos::View<int*, ExecSpace> valid_verts,
  Kokkos::View<bool*, ExecSpace> valid,
  Kokkos::View<int, ExecSpace> max_degree_vert,
  Kokkos::View<double, ExecSpace> avg_degree,
  Kokkos::View<int*, ExecSpace> scc_maps,
  Kokkos::View<int*, ExecSpace> queue,
  Kokkos::View<int*, ExecSpace> queue_next,
  Kokkos::View<int*, ExecSpace> offsets,
  Kokkos::View<int*, ExecSpace> offsets_next,
  Kokkos::View<bool*, ExecSpace> fw,
  Kokkos::View<bool*, ExecSpace> scc)
{  

#if DEBUG
  double elt;
  elt = timer();
#endif

  Kokkos::View<int, ExecSpace> offsets_max_out("o max");
  Kokkos::View<int, ExecSpace> offsets_max_in("i max");    

  if (alg_to_run == 2)
    fwbw_init_offsets<ExecSpace>(n, 
      out_degree_list, in_degree_list,
      fw, scc, max_degree_vert, 
      offsets_max_out, offsets_max_in);
  else
    fwbw_init<ExecSpace>(n, fw, scc, max_degree_vert);

#if DEBUG
  elt = timer() - elt;
  printf("FWBW init: %9.6lf\n", elt);
  elt = timer();
#endif

switch (alg_to_run)
{
  case 0:
    printf("case 0 fwbw_baseline\n");
    fwbw_baseline<ExecSpace>(
      out_degree_list, out_array, 
      valid,
      max_degree_vert, fw,
      queue, queue_next); break;
  case 1:
    printf("case 1 fwbw_manhattan_local\n");
    fwbw_manhattan_local<ExecSpace>(
      out_degree_list, out_array, 
      valid,
      max_degree_vert, fw,
      queue, queue_next); break;
  case 2:
    printf("case 2 fwbw_manhattan_global\n");
    fwbw_manhattan_global<ExecSpace>(
      out_degree_list, out_array, 
      valid,
      max_degree_vert, offsets_max_out, fw,
      queue, queue_next,
      offsets, offsets_next); break;
  default:
    printf("\n MAD FUCKING ERROR \n"); 
    exit(0); break;
}

#if DEBUG
  elt = timer() - elt;
  printf("FWBW fw: %9.6lf\n", elt);
  elt = timer();
#endif

switch (alg_to_run)
{
  case 0:
    fwbw_baseline<ExecSpace>(
      in_degree_list, in_array, 
      fw,
      max_degree_vert, scc,
      queue, queue_next); break;
  case 1:
    fwbw_manhattan_local<ExecSpace>(
      in_degree_list, in_array, 
      fw,
      max_degree_vert, scc,
      queue, queue_next); break;
  case 2:
    fwbw_manhattan_global<ExecSpace>(
      in_degree_list, in_array, 
      fw,
      max_degree_vert, offsets_max_in, scc,
      queue, queue_next,
      offsets, offsets_next); break;
  default:
    printf("\n MAD FUCKING ERROR \n"); 
    exit(0); break;
}

#if DEBUG
  elt = timer() - elt;
  printf("FWBW bw: %9.6lf\n", elt);
  elt = timer();
#endif

  fwbw_update_valid<ExecSpace>(n,
    scc,
    num_valid, valid_verts, valid,
    max_degree_vert,
    scc_maps);

#if DEBUG
  elt = timer() - elt;
  printf("FWBW update: %9.6lf\n", elt);
  elt = timer();
#endif
}
