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

#define TEST 0
#define WARP_SIZE 32

template< class ExecSpace >
struct fwbw_manhattan_global {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef Kokkos::View<long> long_type;

  int_array out_array;
  int_array out_degree_list;
  bool_array valid;

  int num_visited;
  int num_visited_edges;

  int_array queue;
  int_array queue_next;
  int_type queue_size;
  int_array offsets;
  int_array offsets_next;
  long_type sizeq_offsets;
  int_type offsets_max;

  int_type root;
  bool_array visited;
  int_array levels;
  int_type cur_level;

  fwbw_manhattan_global(
    int_array out_degree_list_in, int_array out_array_in,
    bool_array valid_in,
    int_type root_in, int_type root_offset_in, bool_array visited_in,
    int_array queue_in, int_array queue_next_in,
    int_array offsets_in, int_array offsets_next_in)
  : out_array(out_array_in), out_degree_list(out_degree_list_in)
  , valid(valid_in)
  , root(root_in), offsets_max(root_offset_in), visited(visited_in)
  , queue(queue_in), queue_next(queue_next_in)
  , offsets(offsets_in), offsets_next(offsets_next_in)
  , queue_size("queue size") 
  , sizeq_offsets("size q offsets")
  {
    typename int_type::HostMirror host_size = create_mirror(queue_size);
    typename int_type::HostMirror host_offsets_max = create_mirror(offsets_max);
    typename long_type::HostMirror host_sizeq_offsets = create_mirror(sizeq_offsets);
    typename int_type::HostMirror host_root = create_mirror(root);
    deep_copy(host_root, root);
    deep_copy(host_offsets_max, offsets_max);
    host_size() = 1;
    host_sizeq_offsets() = 0;
    num_visited = 1;
    num_visited_edges = 0;
    deep_copy(queue_size, host_size);
    deep_copy(sizeq_offsets, host_sizeq_offsets);

    // initialize the queue with the root node
    typename int_array::HostMirror host_queue = create_mirror_view(queue);
    host_queue(0) = host_root();
    deep_copy(queue, host_queue);

    int team_size = team_policy::team_size_recommended(*this); //best guess
#if DEBUG
    double elt = timer();
#endif
    while (host_size() > 0)
    {
#if DEBUG
      //double level_time = timer();
      printf("%d-- %d %d\n", num_visited, host_size(), host_offsets_max());
#endif
      int num_teams = ((int)host_offsets_max() + WORK_CHUNK - 1 ) / WORK_CHUNK;
      team_policy policy(num_teams, team_size);
      Kokkos::parallel_for(policy , *this);

      deep_copy(host_sizeq_offsets, sizeq_offsets);
      host_size() = (int)((host_sizeq_offsets() >> 32) & 0xFFFFFFFF);   
      host_offsets_max() = (int)(host_sizeq_offsets() & 0xFFFFFFFF);
      num_visited += host_size();
      num_visited_edges += host_offsets_max();
      host_sizeq_offsets() = (long)0;
      deep_copy(queue_size, host_size);
      deep_copy(offsets_max, host_offsets_max);
      deep_copy(sizeq_offsets, host_sizeq_offsets);
#if DO_LEVELS
      *host_cur_level += 1;
      deep_copy(cur_level, host_cur_level);
#endif
      int_array temp = queue;
      queue = queue_next;
      queue_next = temp;
      temp = offsets;
      offsets = offsets_next;
      offsets_next = temp;
#if DEBUG
      //level_time = timer() - level_time;
      //printf("\t%2.6lf\n", level_time);
#endif
    }
  }

  KOKKOS_INLINE_FUNCTION 
  int highest_less_than(
    Kokkos::View<int*, ExecSpace> offsets, 
    int val,
    int bound_low, int bound_high) const
  {
    bool found = false;
    int index = 0;
    while (!found)
    {
      index = (bound_high + bound_low) / 2;
      if (offsets[index] <= val && offsets[index+1] > val)
      {
        found = true;
      }
      else if (offsets[index] <= val)
        bound_low = index;
      else if (offsets[index] > val)
        bound_high = index;
    }

    return index;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev) const
  { 
    if (queue_size() == 1 && queue[0] == root()) 
    {
      offsets[0] = 0;
      offsets[queue_size()] = offsets_max();
    }
    Kokkos::View<int, Kokkos::MemoryUnmanaged> team_queue_size(dev.team_shmem());
    Kokkos::View<int, Kokkos::MemoryUnmanaged> team_sum(dev.team_shmem());
   
    Kokkos::View<long, Kokkos::MemoryUnmanaged> offset_and_sum(dev.team_shmem());
    team_queue_size() = 0;
    team_sum() = 0;
    offset_and_sum() = 0;
    dev.team_barrier();

    int local_buffer[ LOCAL_BUFFER_LENGTH ];
    int local_offsets[ LOCAL_BUFFER_LENGTH ];
    int local_count = 0;
    int local_sum = 0;

    int team_size = dev.team_size();
    int team_rank = dev.team_rank();
    int league_offset = dev.league_rank() * WORK_CHUNK;
    int begin = league_offset + team_rank;
    int end = league_offset + WORK_CHUNK;

    int max_offset = offsets_max();
    int bound_low = 0;
    int bound_high = queue_size();

    if (end < max_offset)
      bound_high = highest_less_than(offsets, min(end, max_offset-1), bound_low, bound_high) + 1;
    
    bool do_search = true;
    int vert; int j;
    for (int i = begin; i < end; i += team_size)
    {
      if (i < max_offset)
      {
        if (do_search)
        {
          bool found = false;
          int new_high = bound_high;
          int new_low = bound_low;
          while (!found)
          {
            j = (new_high + new_low) / 2;
            if (offsets[j] <= i && offsets[j+1] > i)
            {
              found = true;
            }
            else if (offsets[j] <= i)
              new_low = j;
            else if (offsets[j] > i)
              new_high = j;
          }
          bound_low = j;
          vert = queue[j];
        }
        int out = out_vertice(vert, i - offsets[j]);
        if (i + team_size >= offsets[j+1])
          do_search = true;
        else
          do_search = false;

        if (!visited[out] && valid[out])
        {
          visited[out] = true;
          int out_degree = out_degree(out);
          if (out_degree)
          {
            local_buffer[local_count] = out;         
            local_sum += out_degree;
            local_offsets[local_count] = local_sum;
            ++local_count;
          }
        }
      }
    }  


    int team_offset = dev.team_scan(local_count, &team_queue_size());
    int cur_sum = dev.team_scan(local_sum, &team_sum());
    if (team_rank == 0)
    { 
      unsigned long temp = (((long)team_queue_size() << 32) | (long)team_sum());
      offset_and_sum() = Kokkos::atomic_fetch_add(&sizeq_offsets(), temp);
    }
    dev.team_barrier();

    const int queue_offset = (int)((offset_and_sum() >> 32) & 0xFFFFFFFF);
    const int total_offset = (int)(offset_and_sum() & 0xFFFFFFFF);

    team_offset += queue_offset;
    cur_sum += total_offset;

    for (int l = 0; l < local_count; ++l)
      queue_next[team_offset + l] = local_buffer[l];
    ++team_offset;
    for (int l = 0; l < local_count; ++l)
      offsets_next[team_offset + l] = cur_sum + local_offsets[l];
  }

  size_t team_shmem_size(int team_size) const { return sizeof(int)*(256); };
};
