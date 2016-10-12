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
'########:'########::'####:'##::::'##:
... ##..:: ##.... ##:. ##:: ###::'###:
::: ##:::: ##:::: ##:: ##:: ####'####:
::: ##:::: ########::: ##:: ## ### ##:
::: ##:::: ##.. ##:::: ##:: ##. #: ##:
::: ##:::: ##::. ##::: ##:: ##:.:: ##:
::: ##:::: ##:::. ##:'####: ##:::: ##:
:::..:::::..:::::..::....::..:::::..::
*/
template< class ExecSpace >
struct simple_trim {
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_array scc_maps;

  int_type n;
  int_array out_degree_list;
  int_array in_degree_list;

  int_array valid_verts;
  bool_array valid;
  int_type num_valid;

  simple_trim(int_type n_in,
    int_array out_degree_list_in, int_array in_degree_list_in,
    int_type num_valid_in, int_array valid_verts_in,
    bool_array valid_in,
    int_array scc_maps_in) 
  : n(n_in)
  , scc_maps(scc_maps_in) 
  , out_degree_list(out_degree_list_in), in_degree_list(in_degree_list_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in)
  , valid(valid_in)
  {    
    typename int_type::HostMirror host_n = Kokkos::create_mirror_view(n);

    Kokkos::deep_copy(host_n, n);
   
    int num_teams = (*host_n + WORK_CHUNK - 1 ) / WORK_CHUNK;
    int team_size = ExecSpace::team_recommended();
    team_policy policy(num_teams, team_size);
#if DEBUG
    printf("%d -- %d\n", num_teams, team_size);
#endif
    Kokkos::parallel_for(policy , *this);
  }


  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev ) const
  {
    int local_buffer[ LOCAL_BUFFER_LENGTH ];
    int local_count = 0;
    int k = 0;

    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = *n < end ? *n : end;
    int team_size = dev.team_size();

    for (int v = begin; v < end; v += team_size)
    {
      if (out_degree(v) < 1 || in_degree(v) < 1)
      {
        valid[v] = false;
        scc_maps[v] = v;
      }
      else
      {
        valid[v] = true;
        scc_maps[v] = -1;
        local_buffer[local_count++] = v;
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

