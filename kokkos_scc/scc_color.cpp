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
struct color_init {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_array colors;

  int_type num_valid;
  int_array valid_verts;

  int_array queue;
  bool_array in_queue;
  bool_array in_queue_next;

  color_init(int_array colors_in,
    int_type num_valid_in, int_array valid_verts_in,
    int_array queue_in,
    bool_array in_queue_in, bool_array in_queue_next_in)
  : colors(colors_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in)
  , queue(queue_in)
  , in_queue(in_queue_in), in_queue_next(in_queue_next_in)
  {    
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    Kokkos::deep_copy(host_num_valid, num_valid);

    int team_size = ExecSpace::team_recommended();
    int num_teams = (*host_num_valid + WORK_CHUNK - 1 ) / WORK_CHUNK;
    team_policy policy(num_teams, team_size);
    Kokkos::parallel_for(policy , *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const team_member &dev ) const
  {
    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = *num_valid < end ? *num_valid : end;
    int team_size = dev.team_size();

    for (int i = begin; i < end; i += team_size)
    {
      int vert = valid_verts[i];
      colors[vert] = vert;
      in_queue[vert] = true;
      in_queue_next[vert] = false;
      queue[i] = vert;
    }
  }
};

template< class ExecSpace >
struct color_init_offsets {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef Kokkos::View<long, ExecSpace> long_type;

  int_array colors;

  int_type num_valid;
  int_array valid_verts;

  int_array out_degree_list;

  int_array queue;
  int_array offsets;
  bool_array in_queue;
  bool_array in_queue_next;

  long_type sizeq_offsets;
  int_type offsets_max;

  color_init_offsets(int_array colors_in,
    int_array out_degree_list_in,
    int_type num_valid_in, int_array valid_verts_in,
    int_array queue_in, int_array offsets_in, int_type offsets_max_in,
    bool_array in_queue_in, bool_array in_queue_next_in)
  : colors(colors_in)
  , out_degree_list(out_degree_list_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in)
  , queue(queue_in), offsets(offsets_in), offsets_max(offsets_max_in)
  , in_queue(in_queue_in), in_queue_next(in_queue_next_in)
  , sizeq_offsets("size q offsets")
  {    
    typename int_type::HostMirror host_offsets_max = create_mirror(offsets_max);
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    Kokkos::deep_copy(host_num_valid, num_valid);
    typename long_type::HostMirror host_sizeq_offsets = create_mirror(sizeq_offsets);
    *host_sizeq_offsets = 0;
    deep_copy(sizeq_offsets, host_sizeq_offsets);

    int team_size = ExecSpace::team_recommended();
    int num_teams = (*host_num_valid + WORK_CHUNK - 1 ) / WORK_CHUNK;
    team_policy policy(num_teams, team_size);
    Kokkos::parallel_for(policy , *this);

    deep_copy(host_sizeq_offsets, sizeq_offsets);
    *host_offsets_max = (int)(*host_sizeq_offsets & 0xFFFFFFFF);
    deep_copy(offsets_max, host_offsets_max);
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

    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = *num_valid < end ? *num_valid : end;
    int team_size = dev.team_size();

    for (int i = begin; i < end; i += team_size)
    {
      int vert = valid_verts[i];
      colors[vert] = vert;
      in_queue[vert] = true;
      in_queue_next[vert] = false;

      local_buffer[local_count] = vert;        
      local_sum += out_degree(vert);
      local_offsets[local_count] = local_sum;
      ++local_count;
    }

    int team_offset = dev.team_scan(local_count, &team_queue_size());
    int cur_sum = dev.team_scan(local_sum, &team_sum());
    if (dev.team_rank() == 0)
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
      queue[team_offset + l] = local_buffer[l];
    ++team_offset;
    for (int l = 0; l < local_count; ++l)
      offsets[team_offset + l] = cur_sum + local_offsets[l];
  }
  size_t team_shmem_size(int team_size) const { return sizeof(int)*(256); };
};



/*
'########:::'#######:::'#######::'########::'######::
 ##.... ##:'##.... ##:'##.... ##:... ##..::'##... ##:
 ##:::: ##: ##:::: ##: ##:::: ##:::: ##:::: ##:::..::
 ########:: ##:::: ##: ##:::: ##:::: ##::::. ######::
 ##.. ##::: ##:::: ##: ##:::: ##:::: ##:::::..... ##:
 ##::. ##:: ##:::: ##: ##:::: ##:::: ##::::'##::: ##:
 ##:::. ##:. #######::. #######::::: ##::::. ######::
..:::::..:::.......::::.......::::::..::::::......:::
*/
template< class ExecSpace >
struct color_get_roots {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<int, ExecSpace> int_type;

  int_type num_roots;
  int_array roots;

  int_array valid_verts;
  bool_array valid;
  int_type num_valid;

  int_array colors;
  int_array scc_maps;

  color_get_roots(int_array colors_in,
    int_type num_roots_in, int_array roots_in,
    int_type num_valid_in, int_array valid_verts_in,
    bool_array valid_in,
    int_array scc_maps_in)
  : colors(colors_in)
  , num_roots(num_roots_in), roots(roots_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , scc_maps(scc_maps_in)
  {    
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    deep_copy(host_num_valid, num_valid);

    int team_size = ExecSpace::team_recommended();
    int num_teams = (*host_num_valid + WORK_CHUNK - 1 ) / WORK_CHUNK;
    team_policy policy(num_teams, team_size);
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
    end = *num_valid < end ? *num_valid : end;
    int team_size = dev.team_size();

    for (int i = begin; i < end; i += team_size)
    {
      int vert = valid_verts[i];
      if (colors[vert] == vert)
      {
        local_buffer[local_count++] = vert;
        scc_maps[vert] = vert;
        valid[vert] = false;
      }
      ++k;

      if (k == LOCAL_BUFFER_LENGTH)
      {
        const int team_offset = dev.team_scan(local_count, &num_roots());
        for (int l = 0; l < local_count; ++l)
          roots[team_offset+l] = local_buffer[l];
        k = 0;
        local_count = 0;
      }
    }

    const int team_offset = dev.team_scan(local_count, &num_roots());
    for (int l = 0; l < local_count; ++l)
      roots[team_offset+l] = local_buffer[l];
  }
};



template< class ExecSpace >
struct color_get_roots_offsets {
  typedef ExecSpace device_type;

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef Kokkos::View<long, ExecSpace> long_type;

  int_array in_degree_list;

  int_array offsets;
  int_type offsets_max;

  int_type num_roots;
  int_array roots;
  int_array valid_verts;
  bool_array valid;
  int_type num_valid;

  int_array colors;
  int_array scc_maps;

  long_type sizeq_offsets;

  color_get_roots_offsets(int_array colors_in,
    int_array in_degree_list_in,
    int_type num_roots_in, int_array roots_in, 
    int_array offsets_in, int_type offsets_max_in,
    int_type num_valid_in, int_array valid_verts_in, bool_array valid_in,
    int_array scc_maps_in)
  : colors(colors_in)
  , in_degree_list(in_degree_list_in)
  , num_roots(num_roots_in), roots(roots_in)
  , offsets(offsets_in), offsets_max(offsets_max_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in), valid(valid_in)
  , scc_maps(scc_maps_in)
  , sizeq_offsets("size q offsets")
  {    
    typename int_type::HostMirror host_num_roots = create_mirror(num_roots);
    typename int_type::HostMirror host_offsets_max = create_mirror(offsets_max);
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    Kokkos::deep_copy(host_num_valid, num_valid);
    typename long_type::HostMirror host_sizeq_offsets = create_mirror(sizeq_offsets);
    *host_sizeq_offsets = 0;
    deep_copy(sizeq_offsets, host_sizeq_offsets);

    int team_size = ExecSpace::team_recommended();
    int num_teams = (*host_num_valid + WORK_CHUNK - 1 ) / WORK_CHUNK;
    team_policy policy(num_teams, team_size);
    Kokkos::parallel_for(policy , *this);

    deep_copy(host_sizeq_offsets, sizeq_offsets);
    *host_num_roots = (int)((*host_sizeq_offsets >> 32) & 0xFFFFFFFF);
    *host_offsets_max = (int)(*host_sizeq_offsets & 0xFFFFFFFF);
    deep_copy(num_roots, host_num_roots);
    deep_copy(offsets_max, host_offsets_max);
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

    int begin = dev.league_rank() * WORK_CHUNK + dev.team_rank();
    int end = begin + WORK_CHUNK;
    end = *num_valid < end ? *num_valid : end;
    int team_size = dev.team_size();

    for (int i = begin; i < end; i += team_size)
    {
      int vert = valid_verts[i];
      if (colors[vert] == vert)
      {
        valid[vert] = false;
        local_buffer[local_count] = vert;        
        local_sum += in_degree(vert);
        local_offsets[local_count] = local_sum;
        ++local_count;
      }
    }

    int team_offset = dev.team_scan(local_count, &team_queue_size());
    int cur_sum = dev.team_scan(local_sum, &team_sum());
    if (dev.team_rank() == 0)
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
      roots[team_offset + l] = local_buffer[l];
    ++team_offset;
    for (int l = 0; l < local_count; ++l)
      offsets[team_offset + l] = cur_sum + local_offsets[l];
  }
  size_t team_shmem_size(int team_size) const { return sizeof(int)*(256); };
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
struct color_update_valid {

  typedef Kokkos::View<bool*, ExecSpace> bool_array;
  typedef Kokkos::View<int*, ExecSpace> int_array;
  typedef Kokkos::View<unsigned*, ExecSpace> unsigned_array;
  typedef Kokkos::View<int, ExecSpace> int_type;
  typedef ExecSpace device_type;

  int_type n;
  int_array valid_verts;
  bool_array valid;
  int_type num_valid;

  color_update_valid(int_type n_in,
    int_type num_valid_in, int_array valid_verts_in,
    bool_array valid_in) 
  : n(n_in)
  , num_valid(num_valid_in), valid_verts(valid_verts_in)
  , valid(valid_in)
  {    
    typename int_type::HostMirror host_num_valid = create_mirror(num_valid);
    typename int_type::HostMirror host_n = create_mirror(n);
    deep_copy(host_num_valid, num_valid);
    deep_copy(host_n, n);

    *host_num_valid = 0;
    deep_copy(num_valid, host_num_valid);

    int team_size = ExecSpace::team_recommended();
    int num_teams = (*host_n + WORK_CHUNK - 1 ) / WORK_CHUNK;
    team_policy policy(num_teams, team_size);
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

    for (int i = begin; i < end; i += team_size)
    {
      int vert = i;
      if (valid[vert])
        local_buffer[local_count++] = vert;
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
void do_coloring(Kokkos::View<int, ExecSpace> n,
  Kokkos::View<int*, ExecSpace> out_degree_list,
  Kokkos::View<int*, ExecSpace> out_array,
  Kokkos::View<int*, ExecSpace> in_degree_list,
  Kokkos::View<int*, ExecSpace> in_array,
  Kokkos::View<int, ExecSpace> num_valid,
  Kokkos::View<int*, ExecSpace> valid_verts,
  Kokkos::View<bool*, ExecSpace> valid,
  Kokkos::View<int*, ExecSpace> scc_maps,
  Kokkos::View<int*, ExecSpace> queue,
  Kokkos::View<int*, ExecSpace> queue_next,
  Kokkos::View<int*, ExecSpace> offsets,
  Kokkos::View<int*, ExecSpace> offsets_next,
  Kokkos::View<bool*, ExecSpace> in_queue,
  Kokkos::View<bool*, ExecSpace> in_queue_next,
  Kokkos::View<int*, ExecSpace> owner,
  Kokkos::View<int*, ExecSpace> colors)
{
  Kokkos::View<int, ExecSpace> num_roots("roots");  
  Kokkos::View<int, ExecSpace> offsets_max("o max");  
  typename Kokkos::View<int, ExecSpace>::HostMirror host_num_roots = create_mirror(num_roots);
  typename Kokkos::View<int, ExecSpace>::HostMirror host_num_valid = create_mirror(num_valid);

  Kokkos::deep_copy(host_num_valid, num_valid);
  while (*host_num_valid > 0)
  {

#if DEBUG
    double elt;
    elt = timer();
#endif

  if (alg_to_run == 2)
    color_init_offsets<ExecSpace>(colors,
      out_degree_list,
      num_valid, valid_verts, 
      queue, offsets, offsets_max,
      in_queue, in_queue_next);
  else    
    color_init<ExecSpace>(colors,
      num_valid, valid_verts, 
      queue, 
      in_queue, in_queue_next);

#if DEBUG
    elt = timer() - elt;
    printf("Color init: %9.6lf\n", elt);
    elt = timer();
#endif
 
switch (alg_to_run)
{
  case 0:
    color_propagate_baseline<ExecSpace>(colors,
      out_degree_list, out_array,
      num_valid, valid_verts, valid,
      queue, queue_next,
      in_queue, in_queue_next); break;
  case 1:
    color_propagate_manhattan_local<ExecSpace>(colors,
      out_degree_list, out_array,
      num_valid, valid_verts, valid,
      queue, queue_next,
      in_queue, in_queue_next); break;
  case 2:
    color_propagate_manhattan_global<ExecSpace>(colors,
      out_degree_list, out_array,
      num_valid, valid_verts, valid,
      queue, queue_next,
      offsets, offsets_next, offsets_max,
      in_queue, in_queue_next,
      owner); break;
  default:
    printf("\n ERROR - algorithm option incorrect \n"); 
    exit(0); break;
}

#if DEBUG
    elt = timer() - elt;
    printf("Color fw: %9.6lf\n", elt);
    elt = timer();
#endif
  
    host_num_roots = 0;
    Kokkos::deep_copy(num_roots, host_num_roots);

  if (alg_to_run == 2)
    color_get_roots_offsets<ExecSpace>(colors, 
      in_degree_list,
      num_roots, queue, offsets, offsets_max,
      num_valid, valid_verts, valid,
      scc_maps);    
  else
    color_get_roots<ExecSpace>(colors, 
      num_roots, queue,
      num_valid, valid_verts, valid,
      scc_maps);

#if DEBUG
    elt = timer() - elt;
    deep_copy(host_num_roots, num_roots);
    printf("Color roots: %9.6lf, %d\n", elt, *host_num_roots);
    elt = timer();
#endif

switch(alg_to_run)
{
  case 0:
    color_mark_scc_baseline<ExecSpace>(colors,
      num_roots,
      in_degree_list, in_array,
      num_valid, valid_verts, valid,
      scc_maps,
      queue, queue_next); break;
  case 1:
    color_mark_scc_manhattan_local<ExecSpace>(colors,
      num_roots,
      in_degree_list, in_array,
      num_valid, valid_verts, valid,
      scc_maps,
      queue, queue_next); break;
  case 2:
    color_mark_scc_manhattan_global<ExecSpace>(colors,
      num_roots,
      in_degree_list, in_array,
      num_valid, valid_verts, valid,
      scc_maps,
      queue, queue_next,
      offsets, offsets_next, offsets_max); break;
  default:
    printf("\n ERROR - algorithm option incorrect \n"); 
    exit(0); break;
}

#if DEBUG
    elt = timer() - elt;
    printf("Color bw: %9.6lf\n", elt);
    elt = timer();
#endif
  
    color_update_valid<ExecSpace>(n, num_valid, valid_verts, valid);
    Kokkos::deep_copy(host_num_valid, num_valid);

#if DEBUG
    elt = timer() - elt;
    printf("Color update: %9.6lf\n", elt);
    elt = timer();  
    printf("coloring valid: %d\n", *host_num_valid);
#endif
  }
}
