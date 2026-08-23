```@meta
CurrentModule = GaussianBasis
```

# One-Electron Integrals

## Overlap

For the atomic orbitals ``\chi_\mu, \chi_\nu``, the overlap integral is calculated as:

```math
S_{\mu\nu} = \int \chi_\mu(\mathbf{r})\, \chi_\nu(\mathbf{r})\, d\mathbf{r}
```

A full-dense overlap matrix can be calculated directly from a basis set
```julia-repl
julia> bset = BasisSet("sto-3g", """
              O        0.000000000000     -0.143225816552      0.000000000000
              H        1.638036840407      1.136548822547     -0.000000000000
              H       -1.638036840407      1.136548822547     -0.000000000000
              """)
julia> S = overlap(bset)
7×7 Matrix{Float64}:
 1.0         0.236704    0.0        0.0        0.0  0.00410862   0.00410862
 0.236704    1.0         0.0        0.0        0.0  0.0644883    0.0644883
 0.0         0.0         1.0        0.0        0.0  0.0572785   -0.0572785
 0.0         0.0         0.0        1.0        0.0  0.0447509    0.0447509
 0.0         0.0         0.0        0.0        1.0  0.0          0.0
 0.00410862  0.0644883   0.0572785  0.0447509  0.0  1.0          0.0100209
 0.00410862  0.0644883  -0.0572785  0.0447509  0.0  0.0100209    1.0
```
Alternatively, a in-place (mutating) version can be used:

```julia-repl
julia> out = zeros(bset.nbas, bset.nbas)
julia> overlap!(out, bset)
julia> out == S
true
```
A per-shell option is also available:
```julia-repl
julia> S43 = overlap(bset, 4, 3) # 4 and 3 are shell indexes
1×3 Matrix{Float64}:
 0.0572785  0.0447509  0.0
```
For maximum efficiency, you may need to use the most primitive call, mutating and using a pre-allocated output
```julia-repl
julia> out = zeros(num_basis(bset[4]), num_basis(bset[3]))
julia> overlap!(out, bset, 4, 3) 
julia> out == S43
true
```

The overlap can also be computed using two basis set. This will calculate $\langle \chi_\mu | \chi_\nu \rangle$ for $\mu \in A$ and $\nu \in B$, where $A$ and $B$ are two different basis set. This can be useful when projecting results from one basis onto another. 

```julia-repl
julia> bset2 = BasisSet("3-21g", """
              O        0.000000000000     -0.143225816552      0.000000000000
              H        1.638036840407      1.136548822547     -0.000000000000
              H       -1.638036840407      1.136548822547     -0.000000000000
              """);

julia> overlap(bset, bset2)
7×13 Matrix{Float64}:
 0.998058    0.255215   0.0        0.0        0.0       0.207535   …  0.0      3.07065e-6  0.00788467   3.07065e-6   0.00788467
 0.205853    0.853358   0.0        0.0        0.0       0.980775      0.0      0.0105854   0.112372     0.0105854    0.112372
 0.0         0.0        0.886749   0.0        0.0       0.0           0.0      0.0161306   0.0936416   -0.0161306   -0.0936416
 0.0         0.0        0.0        0.886749   0.0       0.0           0.0      0.0126026   0.0731609    0.0126026    0.0731609
 0.0         0.0        0.0        0.0        0.886749  0.0           0.83337  0.0         0.0          0.0          0.0
 0.00354245  0.021765   0.0137339  0.0107301  0.0       0.0800824  …  0.0      0.914077    0.899458     0.00124358   0.0171688
 0.00354245  0.021765  -0.0137339  0.0107301  0.0       0.0800824     0.0      0.00124358  0.0171688    0.914077     0.899458
```


```@docs
overlap
overlap!
```


## Kinetic Energy

For the atomic orbitals ``\chi_\mu, \chi_\nu``, the kinetic energy integral is calculated as:

```math
T_{\mu\nu} = -\frac{1}{2}\int \chi_\mu(\mathbf{r})\, \nabla^2 \chi_\nu(\mathbf{r})\, d\mathbf{r}
```

A full-dense kinetic energy matrix can be calculated directly from a basis set
```julia-repl
julia> bset = BasisSet("sto-3g", """
              O        0.000000000000     -0.143225816552      0.000000000000
              H        1.638036840407      1.136548822547     -0.000000000000
              H       -1.638036840407      1.136548822547     -0.000000000000
              """)
julia> T = kinetic(bset)
7×7 Matrix{Float64}:
 29.0032      -0.168011    0.0          0.0         0.0      -0.00160233  -0.00160233
 -0.168011     0.808128    0.0          0.0         0.0      -0.0167939   -0.0167939
  0.0          0.0         2.52873      0.0         0.0      -0.00612697   0.00612697
  0.0          0.0         0.0          2.52873     0.0      -0.00478691  -0.00478691
  0.0          0.0         0.0          0.0         2.52873   0.0          0.0
 -0.00160233  -0.0167939  -0.00612697  -0.00478691  0.0       0.760032    -0.00447755
 -0.00160233  -0.0167939   0.00612697  -0.00478691  0.0      -0.00447755   0.760032
```
Alternatively, a in-place (mutating) version can be used:

```julia-repl
julia> out = zeros(bset.nbas, bset.nbas)
julia> kinetic!(out, bset)
julia> out == T
true
```
A per-shell option is also available:
```julia-repl
julia> T43 = kinetic(bset, 4, 3) # 4 and 3 are shell indexes
1×3 Matrix{Float64}:
 -0.00612697  -0.00478691  0.0
```
For maximum efficiency, you may need to use the most primitive call, mutating and using a pre-allocated output
```julia-repl
julia> out = zeros(num_basis(bset[4]), num_basis(bset[3]))
julia> kinetic!(out, bset, 4, 3) 
julia> out == T43
true
```

The kinetic energy can also be computed using two basis set. This will calculate $\langle \chi_\mu | -\frac{1}{2}\nabla^2 | \chi_\nu \rangle$ for $\mu \in A$ and $\nu \in B$, where $A$ and $B$ are two different basis set. This can be useful when projecting results from one basis onto another. 

```julia-repl
julia> bset2 = BasisSet("3-21g", """
              O        0.000000000000     -0.143225816552      0.000000000000
              H        1.638036840407      1.136548822547     -0.000000000000
              H       -1.638036840407      1.136548822547     -0.000000000000
              """);

julia> kinetic(bset, bset2)
7×13 Matrix{Float64}:
 29.5083      -2.60308      0.0         …  -4.54054e-5  -0.00358451
 -0.234446     1.16236      0.0            -0.0150019   -0.0147396
  0.0          0.0          3.37591         0.0166165   -0.00997474
  0.0          0.0          0.0            -0.0129822    0.00779312
  0.0          0.0          0.0             0.0          0.0
 -0.00139057  -0.00795767  -0.00295226  …  -0.0013665   -0.00713319
 -0.00139057  -0.00795767   0.00295226      1.03401      0.314867
```


```@docs
kinetic
kinetic!
```


## Nuclear-electron Attraction

For the atomic orbitals ``\chi_\mu, \chi_\nu``, the nuclear attraction integral, summed over every nucleus ``C`` in the molecule with charge ``Z_C`` at position ``\mathbf{R}_C``, is calculated as:

```math
V_{\mu\nu} = -\sum_C Z_C \int \chi_\mu(\mathbf{r}) \frac{1}{|\mathbf{r}-\mathbf{R}_C|} \chi_\nu(\mathbf{r})\, d\mathbf{r}
```

A full-dense nuclear attraction matrix can be calculated directly from a basis set
```julia-repl
julia> bset = BasisSet("sto-3g", """
              O        0.000000000000     -0.143225816552      0.000000000000
              H        1.638036840407      1.136548822547     -0.000000000000
              H       -1.638036840407      1.136548822547     -0.000000000000
              """)
julia> V = nuclear(bset)
7×7 Matrix{Float64}:
 -61.1276      -7.3036      0.0       -0.00405309   0.0      -0.129311   -0.129311
  -7.3036      -9.56106     0.0       -0.0511636    0.0      -0.390434   -0.390434
   0.0          0.0        -9.49705    0.0          0.0      -0.289694    0.289694
  -0.00405309  -0.0511636   0.0       -9.48994      0.0      -0.229019   -0.229019
   0.0          0.0         0.0        0.0         -9.47879   0.0         0.0
  -0.129311    -0.390434   -0.289694  -0.229019     0.0      -3.42421    -0.0375252
  -0.129311    -0.390434    0.289694  -0.229019     0.0      -0.0375252  -3.42421
```
Alternatively, a in-place (mutating) version can be used:

```julia-repl
julia> out = zeros(bset.nbas, bset.nbas)
julia> nuclear!(out, bset)
julia> out == V
true
```
A per-shell option is also available:
```julia-repl
julia> V43 = nuclear(bset, 4, 3) # 4 and 3 are shell indexes
1×3 Matrix{Float64}:
 -0.289694  -0.229019  0.0
```
For maximum efficiency, you may need to use the most primitive call, mutating and using a pre-allocated output
```julia-repl
julia> out = zeros(num_basis(bset[4]), num_basis(bset[3]))
julia> nuclear!(out, bset, 4, 3) 
julia> out == V43
true
```

The nuclear attraction can also be computed using two basis set (the nuclei are always taken from the first `BasisSet`). This will calculate $\langle \chi_\mu | \sum_C -Z_C/|\mathbf{r}-\mathbf{R}_C| | \chi_\nu \rangle$ for $\mu \in A$ and $\nu \in B$, where $A$ and $B$ are two different basis set. This can be useful when projecting results from one basis onto another. 

```julia-repl
julia> bset2 = BasisSet("3-21g", """
              O        0.000000000000     -0.143225816552      0.000000000000
              H        1.638036840407      1.136548822547     -0.000000000000
              H       -1.638036840407      1.136548822547     -0.000000000000
              """);

julia> nuclear(bset, bset2)
7×13 Matrix{Float64}:
 -61.8862      -5.33044      0.0       …  -0.247703   -6.51069e-5  -0.247703
  -6.94425     -9.79189      0.0          -0.706771   -0.0420387   -0.706771
   0.0          0.0        -10.433        -0.499966    0.0630876    0.499966
  -0.00309527  -0.0329122    0.0          -0.395625   -0.0493949   -0.395625
   0.0          0.0          0.0           0.0         0.0          0.0
  -0.122874    -0.21832     -0.114557  …  -2.78587    -0.00443586  -0.0649447
  -0.122874    -0.21832      0.114557     -0.0649447  -3.40551     -2.78587
```


```@docs
nuclear
nuclear!
```


## Multipole Integrals

### Theory

Multipole integrals are AO integrals of Cartesian powers of the position
operator ``\mathbf{r} = (x,y,z)`` about the origin. The dipole, quadrupole,
octupole, and hexadecapole integrals are
```math
\langle \mu | r_a | \nu \rangle, \quad
\langle \mu | r_a r_b | \nu \rangle, \quad
\langle \mu | r_a r_b r_c | \nu \rangle, \quad
\langle \mu | r_a r_b r_c r_d | \nu \rangle
\qquad a,b,c,d \in \{x,y,z\}
```
respectively, each component indexed by which Cartesian direction(s) the
``r`` factor(s) refer to. They are the building blocks for molecular
electric multipole moments (e.g. contracting the dipole integrals with a
density matrix, plus the nuclear point-charge contribution, gives the
molecular dipole moment) and for response properties such as
polarizabilities.

All multipole integrals are evaluated about the coordinate origin -- shift
the molecule's geometry first if a different reference point (e.g. the
center of mass) is needed.

### Usage

```@docs
dipole
dipole!
quadrupole
quadrupole!
octupole
octupole!
hexadecapole
hexadecapole!
```

### Example

```julia-repl
julia> using GaussianBasis, Molecules

julia> water = Molecules.parse_string("""
       O        0.000000000000     -0.143225816552      0.000000000000
       H        1.638036840407      1.136548822547     -0.000000000000
       H       -1.638036840407      1.136548822547     -0.000000000000
       """);

julia> bset = BasisSet("sto-3g", water);

julia> D = dipole(bset);

julia> size(D)
(7, 7, 3)

julia> D[1, 1, :]   # ⟨1s_O|r|1s_O⟩ in atomic units (x, y, z)
3-element Vector{Float64}:
  0.0
 -0.2706575672723027
  0.0
```

`octupole` and `hexadecapole` follow the same convention, with 3 and 4
trailing Cartesian axes respectively.
