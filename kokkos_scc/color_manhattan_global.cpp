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
'########::'########:::'#######::'########::
 ##.... ##: ##.... ##:'##.... ##: ##.... ##:
 ##:::: ##: ##:::: ##: ##:::: ##: ##:::: ##:
 ########:: ########:: ##:::: ##: ########::
 ##.....::: ##.. ##::: ##:::: ##: ##.....:::
 ##:::::::: ##::. ##:: ##:::: ##: ##::::::::
 ##:::::::: ##:::. ##:. #######:: ##::::::::
..:::::::::..:::::..:::.......:::..:::::::::
*/
template< class ExecSpace >
struct color_propagate_manhattan_global {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef Kokkos::View<long, ExecSpace> long_type;
  
  int_type n;
  int_array out_array;
  int_array out_degree_list;
  int_type num_valid;
  int_array valid_verts;
  bool_array valid;

  int num_visited;
  int num_visited_edges;

  int_array colors;

  int_array queue;
  int_array queue_next;
  int_type queue_size;
  int_array offsets;
  int_array offsets_next;
  bool_array in_queue;
  bool_array in_queue_next;
  int_array owner;

  long_type sizeq_offsets;
  int_type offsets_max;

  color_propagate_manhattan_global(int_array colors_in,
    int_array out_degree_list_in, int_array out_array_in,
    int_type num_valid_in, int_array valid_verts_in, bool_array valid_in,
    int_array queue_in, int_array queue_next_in,
    int_array offsets_in, int_array offsets_next_in, 
    int_type offsets_max_in,
    bool_array in_queue_in, bool_array in_queue_next_in,
    int_array owner_in)
  : colors(colors_in)
  , out_array(out_array_in), out_degree_list(out_degree_list_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , queue(queue_in), queue_next(queue_next_in)
  , offsets(offsets_in), offsets_next(offsets_next_in)
  , offsets_max(offsets_max_in)
  , in_queue(in_queue_in), in_queue_next(in_queue_next_in)
  , owner(owner_in)
  , sizeq_offsets("size q offsetss"), queue_size("queue size")
  {
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    typename int_type::HostMirror host_size = create_mirror(queue_size);
    typename int_type::HostMirror host_offsets_max = create_mirror(offsets_max);
    typename long_type::HostMirror host_sizeq_offsets = create_mirror(sizeq_offsets);
    deep_copy(host_num_valid, num_valid);
    deep_copy(host_offsets_max, offsets_max);
    host_size() = host_num_valid();
    host_sizeq_offsets() = 0;
    num_visited = 0;
    num_visited_edges = 0;
    deep_copy(queue_size, host_size);
    deep_copy(sizeq_offsets, host_sizeq_offsets);

    int team_size = team_policy::team_size_recommended(*this);
#if DEBUG
    double elt = timer();
#endif    
    while (host_size() > 0)
    {
#if DEBUG
      printf("%d %d %d\n", host_size(), host_offsets_max(), num_visited_edges);
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

      int_array temp = queue;
      queue = queue_next;
      queue_next = temp;
      temp = offsets;
      offsets = offsets_next;
      offsets_next = temp;
      bool_array in_temp = in_queue;
      in_queue = in_queue_next;
      in_queue_next = in_temp;
    } 
  }

  KOKKOS_INLINE_FUNCTION 
  int highest_less_than(
    Kokkos::View<int*> offsets, 
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
  void operator()( const team_member &dev ) const
  {    
    Kokkos::View<int, Kokkos::MemoryUnmanaged> team_queue_size(dev.team_shmem());
    Kokkos::View<int, Kokkos::MemoryUnmanaged> team_sum(dev.team_shmem());
   
    Kokkos::View<long, Kokkos::MemoryUnmanaged> offset_and_sum(dev.team_shmem());
    team_queue_size() = 0;
    team_sum() = 0;
    offset_and_sum() = 0;
    dev.team_barrier();

    int local_buffer[ LOCAL_BUFFER_LENGTH*2 ];
    int local_offsets[ LOCAL_BUFFER_LENGTH*2 ];
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
        if (i + team_size >= offsets[j+1])
          do_search = true;
        else
          do_search = false;   

        in_queue[vert] = false;
        int color = colors[vert];
        int out = out_vertice(vert, i - offsets[j]); 
        
        if (valid[out])
        {
          int out_color = colors[out];

          if (color > out_color)
          {
            colors[out] = color;

            if (!in_queue_next[out])
            {
              in_queue_next[out] = true;
              int out_degree = out_degree(out);
              if (out_degree)
              {
                local_buffer[local_count] = out;         
                local_sum += out_degree;
                local_offsets[local_count] = local_sum;
                ++local_count;
              }
              //owner[vert] = team_rank;
            }
        
            if (!in_queue_next[vert])// && owner[vert] == team_rank)
            {
              in_queue_next[vert] = true;
              //owner[vert] = -1;
              local_buffer[local_count] = vert;
              local_sum += out_degree(vert);
              local_offsets[local_count] = local_sum;
              ++local_count;
            }
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





template< class ExecSpace >
struct color_mark_scc_manhattan_global {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef Kokkos::View<long, ExecSpace> long_type;

  int_array in_array;
  int_array in_degree_list;

  int_type num_valid;
  int_array valid_verts;
  bool_array valid;

  int_array queue;
  int_array queue_next;
  int_array offsets;
  int_array offsets_next;
  int_type queue_size;
  int_type next_size;
  bool_array in_queue;
  bool_array in_queue_next;

  int_array colors;
  int_array scc_maps;
  int_type num_roots;
  int_type offsets_max;
  long_type sizeq_offsets;

  color_mark_scc_manhattan_global(int_array colors_in, 
    int_type num_roots_in,
    int_array in_degree_list_in, int_array in_array_in,
    int_type num_valid_in, int_array valid_verts_in, bool_array valid_in,
    int_array scc_maps_in,
    int_array queue_in, int_array queue_next_in,
    int_array offsets_in, int_array offsets_next_in, 
    int_type offsets_max_in)
  : colors(colors_in)
  , num_roots(num_roots_in)
  , in_degree_list(in_degree_list_in), in_array(in_array_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , scc_maps(scc_maps_in)
  , queue(queue_in), queue_next(queue_next_in)
  , offsets(offsets_in), offsets_next(offsets_next_in)
  , offsets_max(offsets_max_in)
  , queue_size("queue size"), next_size("next size")
  , sizeq_offsets("size q offsets")
  {
    typename int_type::HostMirror host_size = create_mirror(queue_size);
    typename int_type::HostMirror host_next = create_mirror(next_size);
    typename int_type::HostMirror host_offsets_max = create_mirror(offsets_max);
    typename long_type::HostMirror host_sizeq_offsets = create_mirror(sizeq_offsets);
    deep_copy(host_size, num_roots);
    deep_copy(queue_size, host_size);
    deep_copy(host_offsets_max, offsets_max);
    int num_visited = 0;
    int num_visited_edges = 0;

    while (host_size() > 0)
    {
#if DEBUG
      printf("%d %d\n", host_size(), host_offsets_max());
#endif
      int team_size = team_policy::team_size_recommended(*this);
      int num_teams = (host_offsets_max() + team_size - 1 ) / team_size;
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

      int_array temp = queue;
      queue = queue_next;
      queue_next = temp;
      temp = offsets;
      offsets = offsets_next;
      offsets_next = temp;
    }
  }

  KOKKOS_INLINE_FUNCTION 
  int highest_less_than(
    Kokkos::View<int*> offsets, 
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
  void operator()( const team_member &dev ) const
  {
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
    int vert; int j = 0; int color = -1;
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
          color = colors[vert];
        }
        if (i + team_size >= offsets[j+1])
          do_search = true;
        else
          do_search = false;

        int in = in_vertice(vert, i - offsets[j]);  
        int in_color = colors[in];    
        if (valid[in] && color == in_color)
        {
          valid[in] = false;
          scc_maps[in] = color;
          int in_degree = in_degree(in);
          if (in_degree)
          {
            local_buffer[local_count] = in;         
            local_sum += in_degree;
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
