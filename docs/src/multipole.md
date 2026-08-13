```@meta
CurrentModule = GaussianBasis
```

# Multipole Integrals

## Theory

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

## Usage

```@docs
dipole(::BasisSet)
dipole(::BasisSet, ::Any, ::Any)
quadrupole(::BasisSet)
quadrupole(::BasisSet, ::Any, ::Any)
GaussianBasis.octupole(::BasisSet)
GaussianBasis.octupole(::BasisSet, ::Any, ::Any)
GaussianBasis.hexadecapole(::BasisSet)
GaussianBasis.hexadecapole(::BasisSet, ::Any, ::Any)
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

julia> D = dipole(bset);

julia> size(D)
(7, 7, 3)

julia> D[1, 1, :]   # ⟨1s_O|r|1s_O⟩ in atomic units (x, y, z)
3-element Vector{Float64}:
  0.0
 -0.2706575672723027
  0.0
```

`octupole` and `hexadecapole` are not exported (call as
`GaussianBasis.octupole`/`GaussianBasis.hexadecapole`) but follow the same
convention, with 3 and 4 trailing Cartesian axes respectively.
