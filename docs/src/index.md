```@meta
CurrentModule = GaussianBasis
```

# GaussianBasis.jl

[GaussianBasis.jl](https://github.com/FermiQC/GaussianBasis.jl) evaluates
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
julia> using GaussianBasis

julia> bset = BasisSet("sto-3g", """
       O        0.000000000000     -0.143225816552      0.000000000000
       H        1.638036840407      1.136548822547     -0.000000000000
       H       -1.638036840407      1.136548822547     -0.000000000000
       """);

sto-3g Basis Set
Type: Spherical   Backend: Libcint

Number of shells: 5
Number of basis:  7

O: 1s 2s 1p
H: 1s
H: 1s
```

Geometries are Cartesian coordinates in Angstrom. From a `BasisSet`, you can compute:

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
7×7 Matrix{Float64}:
 1.0         0.236704    0.0        0.0        0.0  0.00410862   0.00410862
 0.236704    1.0         0.0        0.0        0.0  0.0644883    0.0644883
 0.0         0.0         1.0        0.0        0.0  0.0572785   -0.0572785
 0.0         0.0         0.0        1.0        0.0  0.0447509    0.0447509
 0.0         0.0         0.0        0.0        1.0  0.0          0.0
 0.00410862  0.0644883   0.0572785  0.0447509  0.0  1.0          0.0100209
 0.00410862  0.0644883  -0.0572785  0.0447509  0.0  0.0100209    1.0
```

## Basis sets and Shell functions

```@docs
BasisSet
SphericalShell
CartesianShell
GaussianBasis.atomic_orbital_amplitude
GaussianBasis.Libcint
```
