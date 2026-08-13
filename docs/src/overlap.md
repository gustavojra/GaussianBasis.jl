```@meta
CurrentModule = GaussianBasis
```

# One-Electron Integrals

## Theory

For AOs ``\chi_\mu, \chi_\nu``, GaussianBasis.jl evaluates three standard
one-electron integrals:

**Overlap**
```math
S_{\mu\nu} = \int \chi_\mu(\mathbf{r})\, \chi_\nu(\mathbf{r})\, d\mathbf{r}
```

**Kinetic energy**
```math
T_{\mu\nu} = -\frac{1}{2}\int \chi_\mu(\mathbf{r})\, \nabla^2 \chi_\nu(\mathbf{r})\, d\mathbf{r}
```

**Nuclear attraction**, summed over every nucleus ``C`` in the molecule with
charge ``Z_C`` at position ``\mathbf{R}_C``
```math
V_{\mu\nu} = -\sum_C Z_C \int \chi_\mu(\mathbf{r}) \frac{1}{|\mathbf{r}-\mathbf{R}_C|} \chi_\nu(\mathbf{r})\, d\mathbf{r}
```

Together, ``S``, ``T``, and ``V`` are exactly the one-electron pieces needed
to assemble a core Hamiltonian, ``H_\text{core} = T + V``, for Hartree-Fock
or any other SCF method.

Each also has a **mixed-basis** form, ``S_{\mu\nu}`` etc. with ``\chi_\mu``
drawn from one `BasisSet` and ``\chi_\nu`` from another -- useful for e.g.
projecting a density or orbital coefficients from one basis onto another.

## Usage

```@docs
overlap(::BasisSet)
overlap(::BasisSet, ::Any, ::Any)
overlap(::BasisSet, ::BasisSet)
overlap(::BasisSet, ::BasisSet, ::Any, ::Any)
kinetic(::BasisSet)
kinetic(::BasisSet, ::Any, ::Any)
kinetic(::BasisSet, ::BasisSet)
kinetic(::BasisSet, ::BasisSet, ::Any, ::Any)
nuclear(::BasisSet)
nuclear(::BasisSet, ::Any, ::Any)
nuclear(::BasisSet, ::BasisSet)
nuclear(::BasisSet, ::BasisSet, ::Any, ::Any)
```

## Example

```julia-repl
julia> using GaussianBasis, Molecules

julia> water = Molecules.parse_string("""
       O        0.000000000000     -0.143225816552      0.000000000000
       H        1.638036840407      1.136548822547     -0.000000000000
       H       -1.638036840407      1.136548822547     -0.000000000000
       """);

julia> bset = BasisSet("sto-3g", water);

julia> S = overlap(bset);

julia> S[1:3, 1:3]
3×3 Matrix{Float64}:
 1.0     0.2367  0.0
 0.2367  1.0     0.0
 0.0     0.0     1.0

julia> T = kinetic(bset);

julia> T[1, 1]
29.003204213418936

julia> V = nuclear(bset);

julia> V[1, 1]
-61.12760191532953
```

For repeated calls (e.g. across a geometry scan), use the mutating `!`
forms (`overlap!`, `kinetic!`, `nuclear!`) to write into a preallocated
`nbas × nbas` array instead of reallocating each time.
