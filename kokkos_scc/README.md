# Strongly connected components using Kokkos

Contact: George M. Slota -- slotag@rpi.edu

## Status

29 August 2017: 
Successfully tested on a IBM Power8, Nvidia P100, and Nvidia K80 with OpenMP and Cuda.

## To make and run

1. Get Kokkos repo (`git clone https://github.com/kokkos/kokkos.git`)

2. Use Make by modifying `Connectivity/kokkos_scc/Makefile` or use CMake (details omitted)
    1. Set KOKKOS_PATH to base path of kokkos repo
    2. Set KOKKOS_DEVICES and KOKKOS_ARCH for GPU and/or CPU architecture, if needed (see Kokkos documentation)

3. For CPU compilation: `Connectivity/kokkos_scc$ make -j KOKKOS_DEVICES=OpenMP`
3.a For GPU compilation: `Connectivity/kokkos_scc$ make -j KOKKOS_DEVICES=Cuda`

4. To run: `./scc_main.host [graphfile] [algorithm]`

    [graphfile] has format:
    ```
    n m
    v0 v1
    v0 v2
    v1 v2
    ....
    ```

    * n is number of vertices
    * m is number of edges
    * subsequent lines are directed edges from vertex id to vertex id
    * vertices are expected to be 0-indexed
    * the maximum vertex id should be (n-1)
    * the number of lines in [graphfile] should be (m+1)

    [algorithm] choices:
    * 0 - baseline parallelism
    * 1 - local manhattan collapse
    * 2 - global manhattan collapse

## Additional explanation and discussion 

```
@inproceedings{slota_ipdps2015,
  author    = {G. M. Slota and S. Rajamanickam and K. Madduri},
  title     = {High-performance Graph Analytics on Manycore Processors},
  booktitle = {International Parallel \& Distributed Processing Symposium ({IPDPS})},
  year      = {2015}
}
```
