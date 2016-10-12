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
:'######:::'#######::'##::::::::'#######::'########::
'##... ##:'##.... ##: ##:::::::'##.... ##: ##.... ##:
 ##:::..:: ##:::: ##: ##::::::: ##:::: ##: ##:::: ##:
 ##::::::: ##:::: ##: ##::::::: ##:::: ##: ########::
 ##::::::: ##:::: ##: ##::::::: ##:::: ##: ##.. ##:::
 ##::: ##: ##:::: ##: ##::::::: ##:::: ##: ##::. ##::
. ######::. #######:: ########:. #######:: ##:::. ##:
:......::::.......:::........:::.......:::..:::::..::
*/
template< class ExecSpace >
struct color_propagate_baseline {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_array out_array;
  int_array out_degree_list;

  int_array colors;

  int_array queue;
  int_array queue_next;
  int_type queue_size;
  int_type next_size;
  bool_array in_queue;
  bool_array in_queue_next;

  int_type num_valid;
  int_array valid_verts;
  bool_array valid;

  color_propagate_baseline (int_array colors_in,
    int_array out_degree_list_in, int_array out_array_in,
    int_type num_valid_in, int_array valid_verts_in, bool_array valid_in,
    int_array queue_in, int_array queue_next_in,
    bool_array in_queue_in, bool_array in_queue_next_in)
  : colors(colors_in)
  , out_array(out_array_in), out_degree_list(out_degree_list_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , queue(queue_in), queue_next(queue_next_in)
  , in_queue(in_queue_in), in_queue_next(in_queue_next_in)
  , queue_size("queue size"), next_size("next size")
  {
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    typename int_type::HostMirror host_size = create_mirror(queue_size);
    typename int_type::HostMirror host_next = create_mirror(next_size);
    deep_copy(host_num_valid, num_valid);
    deep_copy(queue_size, host_num_valid);
    deep_copy(host_size, host_num_valid);

    int team_size = ExecSpace::team_recommended();
#if DEBUG
    double elt = timer();
#endif    
    while (*host_size > 0)
    {
#if DEBUG
      printf("%d\n", *host_size);
#endif
      int num_teams = ( *host_size + WORK_CHUNK - 1 ) / WORK_CHUNK;
      team_policy policy(num_teams, team_size);
      Kokkos::parallel_for(policy , *this);

      deep_copy(host_next, next_size);
      deep_copy(queue_size, next_size);
      *host_size = *host_next;
      *host_next = 0;
      deep_copy(next_size, host_next); 

      int_array temp = queue;
      queue = queue_next;
      queue_next = temp;
      bool_array in_temp = in_queue;
      in_queue = in_queue_next;
      in_queue_next = in_temp;
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev ) const
  {
    int local_buffer[ LOCAL_BUFFER_LENGTH ];
    int local_count = 0;

    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    int team_size = dev.team_size();
  
    for (int i = begin; i < end; i += team_size)
    {
      int out_degree = 0;
      int vert = -1;
      int color = -1;
      bool changed = false;

      if (i < *queue_size)
      {
        vert = queue[i];
        in_queue[vert] = false;
        color = colors[vert];
        out_degree = out_degree(vert);
      }

      for (int j = 0; j < out_degree; ++j)
      {
        int out = out_vertice(vert, j);
        int out_color = colors[out];

        if (color > out_color)
        {
          colors[out] = color;
          changed = true;

          if (!in_queue_next[out])
          {
            in_queue_next[out] = true;
            local_buffer[local_count++] = out;
          }
        }

        if (local_count == LOCAL_BUFFER_LENGTH)
        {    
          const int thread_start = Kokkos::atomic_fetch_add( 
            &next_size(), local_count);
          for (int l = 0; l < local_count; ++l)
            queue_next[thread_start+l] = local_buffer[l];          
          local_count = 0;
        }
      }

      if (changed && !in_queue_next[vert])
      {
        in_queue_next[vert] = true;
        local_buffer[local_count++] = vert;
      }          

      if (local_count == LOCAL_BUFFER_LENGTH)
      {    
        const int thread_start = Kokkos::atomic_fetch_add( 
          &next_size(), local_count);
        for (int l = 0; l < local_count; ++l)
          queue_next[thread_start+l] = local_buffer[l];          
        local_count = 0;
      }
    }   

    const int team_offset = dev.team_scan(local_count, &next_size());
    for (int l = 0; l < local_count; ++l)
      queue_next[team_offset+l] = local_buffer[l];
  }

};




template< class ExecSpace >
struct color_mark_scc_baseline {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_array in_array;
  int_array in_degree_list;

  int_type num_valid;
  int_array valid_verts;
  bool_array valid;

  int_array queue;
  int_array queue_next;
  int_type queue_size;
  int_type next_size;

  int_array colors;
  int_array scc_maps;
  int_type num_roots;

  color_mark_scc_baseline(int_array colors_in,
    int_type num_roots_in,
    int_array in_degree_list_in, int_array in_array_in,
    int_type num_valid_in, int_array valid_verts_in, bool_array valid_in, 
    int_array scc_maps_in,
    int_array queue_in, int_array queue_next_in)
  : colors(colors_in)
  , num_roots(num_roots_in)
  , in_degree_list(in_degree_list_in), in_array(in_array_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , scc_maps(scc_maps_in), queue(queue_in), queue_next(queue_next_in)
  , queue_size("queue size"), next_size("next size")
  {
    typename int_type::HostMirror host_size = create_mirror(queue_size);
    typename int_type::HostMirror host_next = create_mirror(next_size);
    deep_copy(queue_size, num_roots);
    deep_copy(host_size, num_roots);

    while (*host_size > 0)
    {
      int team_size = ExecSpace::team_recommended();
      int num_teams = (*host_size + WORK_CHUNK - 1 ) / WORK_CHUNK;
      team_policy policy(num_teams, team_size);
      Kokkos::parallel_for(policy , *this);

      deep_copy(host_next, next_size);
      deep_copy(queue_size, next_size);
      *host_size = *host_next;
      *host_next = 0;
      deep_copy(next_size, host_next); 

      int_array temp = queue;
      queue = queue_next;
      queue_next = temp;
#if DEBUG
      printf("%d\n", *host_size);
#endif
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev ) const
  {
    int local_buffer[ LOCAL_BUFFER_LENGTH ];
    int local_count = 0;
    int k = 0;

    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = *queue_size < end ? *queue_size : end;
    int team_size = dev.team_size();
  
    for (int i = begin; i < end; i += team_size)
    {
      int vert = queue[i];
      int color = colors[vert];

      int in_degree = in_degree(vert);
      for (int j = 0; j < in_degree; ++j)
      {
        int in = in_vertice(vert, j);
        int in_color = colors[in];

        if (valid[in] && color == in_color)
        {
          valid[in] = false;
          scc_maps[in] = color;
          local_buffer[local_count++] = in;
        }          
        ++k;

        if (k == LOCAL_BUFFER_LENGTH)
        {
          const int thread_start = Kokkos::atomic_fetch_add( 
            &next_size(), local_count);            
          for (int l = 0; l < local_count; ++l)
            queue_next[thread_start+l] = local_buffer[l];
          
          local_count = 0;
          k = 0;
        }
      }
    }   

    const int team_offset = dev.team_scan(local_count, &next_size());
    for (int l = 0; l < local_count; ++l)
      queue_next[team_offset+l] = local_buffer[l];
  }

};
