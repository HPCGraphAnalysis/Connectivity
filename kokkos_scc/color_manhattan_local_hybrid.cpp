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
'########:'##:::::'##:::::::::::'######:::'######:::'######::
 ##.....:: ##:'##: ##::::::::::'##... ##:'##... ##:'##... ##:
 ##::::::: ##: ##: ##:::::::::: ##:::..:: ##:::..:: ##:::..::
 ######::: ##: ##: ##:'#######:. ######:: ##::::::: ##:::::::
 ##...:::: ##: ##: ##:........::..... ##: ##::::::: ##:::::::
 ##::::::: ##: ##: ##::::::::::'##::: ##: ##::: ##: ##::: ##:
 ##:::::::. ###. ###:::::::::::. ######::. ######::. ######::
..:::::::::...::...:::::::::::::......::::......::::......:::
*/
template< class ExecSpace >
struct fwbw_manhattan_local_hybrid {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef Kokkos::View<double, ExecSpace> double_type;

  int_type root;
  double_type avg_degree;

  int_array out_array;
  int_array out_degree_list;
  int_array in_array;
  int_array in_degree_list;

  int_type num_valid,
  int_array valid_verts;
  bool_array valid;

  int num_visited;
  int_type root;

  int_array queue;
  int_array queue_next;
  int_type queue_size;
  int_type next_size;

  bool_array visited;

  fwbw_manhattan_local_hybrid(
    int_array out_degree_list_in, int_array out_array_in,
    int_array in_degree_list_in, int_array in_array_in,
    int_type num_valid_in, int_array valid_verts_in, bool_array valid_in,
    int_type root_in, double_type avg_degree_in, bool_array visited_in,
    int_array queue_in, int_array queue_next_in)
  : out_array(out_array_in), out_degree_list(out_degree_list_in)
  , in_array(in_array_in), in_degree_list(in_degree_list_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , root(root_in), avg_degree(avg_degree_in), visited(visited_in)
  , queue(queue_in), queue_next(queue_next_in)
  , queue_size("queue size"), next_size("next size")
  {
    typename int_type::HostMirror host_size = create_mirror(queue_size);
    typename int_type::HostMirror host_next = create_mirror(next_size);
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    typename int_type::HostMirror host_root = create_mirror(root);
    typename double_type::HostMirror host_avg_deg = create_mirror(avg_degree);
    *host_size = 1;
    deep_copy(host_root, root);
    deep_copy(host_avg_deg, avg_degree);
    deep_copy(queue, host_root);
    deep_copy(host_num_valid, num_valid);
    deep_copy(queue_size, host_num_valid);

    hybrid = false;
    done_switch = false;
    num_fw = 1;

    int team_size = team_policy::team_size_recommended(*this);
    while (*host_size > 0)
    {
#if DEBUG
      printf("%d %d\n", num_fw, *host_size);
#endif
      int num_teams;
      if (hybrid)
        num_teams = ( *host_num_valid + WORK_CHUNK - 1 ) / WORK_CHUNK;
      else
        num_teams = ( *host_size + team_size - 1) / team_size;

      team_policy policy(num_teams, team_size);
      Kokkos::parallel_for(policy , *this);

      deep_copy(host_next, next_size);
      num_fw += *host_next;
      if (!done_switch && !hybrid)
      { 
        double edges_frontier = (double)*host_next * *host_avg_deg;
        double edges_remainder = (double)(*host_num_valid - num_fw) * *host_avg_deg;

        if ((edges_remainder / ALPHA) < edges_frontier)
        {
          hybrid = true;
#if DEBUG
          printf("\tswitching to hybrid\n");
#endif
        }
      }
      else if (!done_switch && (*host_num_valid / BETA) > *host_next)
      {
        hybrid = false;
        done_switch = true;
#if DEBUG
        printf("\tswitching back\n");
#endif
      }
      *host_size = *host_next;
      if (!hybrid)
      {
        deep_copy(queue_size, host_size);
        int_array temp = queue;
        queue = queue_next;
        queue_next = temp;
      }
      else
      {
        deep_copy(queue_size, host_num_valid);
      }
      *host_next = 0;
      deep_copy(next_size, host_next);
    }
#if DEBUG
  typename bool_array::HostMirror host_fw = create_mirror(fw);
  deep_copy(host_fw, fw);
  int count = 0;
  for (int i = 0; i < host_fw.size(); ++i)
    if (host_fw[i])
      ++count;
  printf("FW count: %d\n", count);
#endif
  }


  KOKKOS_INLINE_FUNCTION 
  int highest_less_than(
    Kokkos::View<int*, Kokkos::MemoryUnmanaged> tripcnts,
    int val, int bound_low) const
  {
    bool found = false;
    int index = 0;
    int bound_high = tripcnts.size()-1;
    while (!found)
    {
      index = (bound_high + bound_low) / 2;
      if (tripcnts[index] <= val && tripcnts[index+1] > val)
      {
        return index;
      }
      else if (tripcnts[index] <= val)
        bound_low = index;
      else if (tripcnts[index] > val)
        bound_high = index;
    }

    return index;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev) const
  {
    int local_buffer[ LOCAL_BUFFER_LENGTH ];
    int local_count = 0;
    int k = 0;

    int team_size = dev.team_size();

    if (!hybrid)
    {
      int team_rank = dev.team_rank();
      int league_offset = dev.league_rank() * team_size;
      int index = league_offset + team_rank;
      int out_degree;

      if (index < queue_size())
        out_degree = out_degree(queue[index]);
      else
        out_degree = 0;
      
      Kokkos::View<int*, Kokkos::MemoryUnmanaged> tripcnts(dev.team_shmem(), team_size+1);
      Kokkos::View<int, Kokkos::MemoryUnmanaged> tripcnt(dev.team_shmem());
      tripcnt() = 0;
      int prefix_sum = dev.team_scan(out_degree, &tripcnt());
      tripcnts[team_rank+1] = prefix_sum + out_degree;
      tripcnts[0] = 0;

      int local_buffer[ LOCAL_BUFFER_LENGTH ];
      int local_count = 0;
      int k = 0;

      dev.team_barrier();
      int begin = team_rank;
      int end = team_size * (tripcnt() / team_size + 1);
      bool do_search = true;
      int vert; int j = 0;
      for (int i = begin; i < end; i += team_size)
      {
        if (i < tripcnt())
        {
          if (do_search)
          {
            j = highest_less_than(tripcnts, i, j);
            vert = queue[league_offset + j];
          }
          int out = out_vertice(vert, i - tripcnts[j]);          
          if (i + team_size > tripcnts[j+1])
            do_search = true;
          else
            do_search = false;

          if (!visited[out] && valid[out])
          {
            visited[out] = true;     
            local_buffer[local_count++] = out;
          }
        }
        ++k;

        if (k == LOCAL_BUFFER_LENGTH)
        {
          const int team_offset = dev.team_scan(local_count, &next_size());
          for (int l = 0; l < local_count; ++l)
            queue_next[team_offset+l] = local_buffer[l];
          local_count = 0;
          k = 0;
        }
      }

    }
    else
    {    
      int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
      int end = begin + WORK_CHUNK;
      for (int i = begin; i < end; i += team_size)
      {
        if (i < queue_size())
        {
          int vert = valid_verts[i];
          if (!fw[vert] && valid[vert])
          {
            int in_degree = in_degree(vert);
            for (int j = 0; j < in_degree; ++j)
            {
              int in = in_vertice(vert, j);
              if (fw[in] && valid[in])
              {
                fw[vert] = true;
                local_buffer[local_count++] = vert;
                break;
              }
            }
          }
        }
        ++k;

        if (k == LOCAL_BUFFER_LENGTH)
        {
          const int team_offset = dev.team_scan(local_count, &next_size());
          for (int l = 0; l < local_count; ++l)
            queue_next[team_offset+l] = local_buffer[l];
          local_count = 0;
          k = 0;
        }
      }
    }

    const int team_offset = dev.team_scan(local_count, &next_size());
    for (int l = 0; l < local_count; ++l)
      queue_next[team_offset+l] = local_buffer[l];
  }

  size_t team_shmem_size(int team_size) const { return sizeof(int)*(team_size*2); };
};
