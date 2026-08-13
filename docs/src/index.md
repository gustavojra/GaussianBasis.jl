```@meta
CurrentModule = GaussianBasis
```

# GaussianBasis.jl

[GaussianBasis.jl](https://github.com/gustavojra/GaussianBasis.jl) evaluates
molecular integrals (and their nuclear-coordinate derivatives) over
contracted Gaussian-type atomic orbitals, using [libcint](https://github.com/sunqm/libcint)
as its default backend. It is the integral engine behind
[Fermi.jl](https://github.com/FermiQC/Fermi.jl).

## Installation

```julia
using Pkg
Pkg.add("GaussianBasis")
```

## Quickstart

A [`BasisSet`](@ref) pairs a set of atoms with a set of contracted Gaussian
basis functions read from a standard basis set library (e.g. `sto-3g`,
`cc-pvdz`):

```julia-repl
julia> using GaussianBasis, Molecules

julia> water = Molecules.parse_string("""
       O        0.000000000000     -0.143225816552      0.000000000000
       H        1.638036840407      1.136548822547     -0.000000000000
       H       -1.638036840407      1.136548822547     -0.000000000000
       """);

julia> bset = BasisSet("sto-3g", water)
sto-3g Basis Set
Number of shells: 5
Number of basis:  7

O: 1s 2s 1p
H: 1s
H: 1s
```

Geometries are Cartesian coordinates in Angstrom; internally, integrals are
evaluated in atomic units (bohr). From a `BasisSet`, you can compute:

  - one-electron integrals (overlap, kinetic, nuclear attraction) -- see
    [One-Electron Integrals](@ref)
  - two-electron integrals (2-, 3-, and 4-center ERIs) -- see
    [Two-Electron Integrals](@ref)
  - multipole integrals (dipole, quadrupole, ...) -- see
    [Multipole Integrals](@ref)
  - nuclear-coordinate gradients and Hessians of all of the above -- see
    [Gradients](@ref) and [Hessians](@ref)

```julia-repl
julia> S = overlap(bset);

julia> size(S)
(7, 7)
```

## Basis sets and library structure

```@docs
BasisSet
SphericalShell
CartesianShell
GaussianBasis.atomic_orbital_amplitude
GaussianBasis.Libcint
```
