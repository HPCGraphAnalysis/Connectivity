Strongly connected components using Kokkos

## Status 
7/28/16:  This code will not compile with the current version of Trilinos;
it will need updates in order to compile.

8/21/17:  Updated to use Kokkos 2.03.13.  Runs successfully on the google web
graph here:
`
http://www.cs.rpi.edu/~slotag/classes/FA16/data/google.graph

8/24/17: Initial GPU support.

On the google web graph it runs successfully for algorithm=0 on a P100 GPU
, but finds an incorrect Max SCC for algorithm=[1|2]:

```
==> google_alg0.out <==
Num SCCs: 371764, Nontrivial: 12874, Max SCC: 434818, Unassigned 0
Done,  0.429645

==> google_alg1.out <==
Num SCCs: 371774, Nontrivial: 12869, Max SCC: 435057, Unassigned 0
Done,  0.532812

==> google_alg2.out <==
Num SCCs: 310304, Nontrivial: 8697, Max SCC: 435032, Unassigned 74459
Done,  0.623173
```

Similarly, on a Power8 host, algorithm=[1|2] finds the correct Max SCC, but
algorithm=2 is incorrect:

```
==> google_alg0.out <==
Num SCCs: 371764, Nontrivial: 12874, Max SCC: 434818, Unassigned 0
Done,  1.201805

==> google_alg1.out <==
Num SCCs: 371764, Nontrivial: 12874, Max SCC: 434818, Unassigned 0
Done,  1.415422

==> google_alg2.out <==
Num SCCs: 303071, Nontrivial: 3396, Max SCC: 478290, Unassigned 80172
Done,  1.384869
```

8/28/2017: GPU support fixed for manhattan local, global still appears to have
color propagation issues.

8/29/2017: GPU support fixed for manhattan global.
