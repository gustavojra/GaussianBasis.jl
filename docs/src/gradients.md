```@meta
CurrentModule = GaussianBasis
```

# Gradients

## Theory

Analytic nuclear gradients of an integral are its first derivative with
respect to the Cartesian coordinates of a single atom `A`, e.g.
```math
\frac{\partial S_{\mu\nu}}{\partial \mathbf{R}_A}, \qquad
\frac{\partial (\mu\nu|\lambda\sigma)}{\partial \mathbf{R}_A}
```
returned as an extra length-3 trailing axis (``x,y,z``). These are the
building blocks of analytic energy gradients (e.g. Hartree-Fock or
post-HF forces) and CPHF/CPKS response equations, avoiding the cost and
numerical noise of finite-difference geometry displacement.

!!! warning "Units: `R_A` is in bohr, not Angstrom"
    A [`BasisSet`](@ref) stores and accepts nuclear coordinates in Angstrom
    (see [Quickstart](@ref)), but every gradient/Hessian in this package is
    computed and returned with respect to bohr displacement of the nucleus,
    since that's the unit libcint (and the integrals themselves) work in
    internally. So `∇overlap(bset, iA)` gives `∂S/∂R_iA` in units of
    (whatever `S`'s unit is) per bohr, *not* per Angstrom, even though
    `bset.atoms[iA].xyz` is in Angstrom. To convert to a per-Angstrom
    derivative, divide by `Molecules.bohr_to_angstrom` (≈0.529177 Å/bohr),
    i.e. multiply by ≈1.8897 bohr/Å.

An integral's derivative is exactly zero, without any computation, whenever
every AO shell contributing to it -- and, for nuclear attraction, every
nucleus supplying the potential -- sits on some atom *other than* `A`
(translational invariance: nothing in the integral depends on `A`'s
position), or when every contributing shell/nucleus sits on `A` itself and
there is no external reference point to move relative to (the "all on `A`"
case is instead recovered from the other atoms' gradients via that same
invariance, ``\sum_A \partial X/\partial \mathbf{R}_A = 0``).

## Usage

```@docs
∇overlap(::BasisSet, ::Any)
∇kinetic(::BasisSet, ::Any)
∇nuclear(::BasisSet, ::Any)
∇ERI_2e4c(::BasisSet, ::Any)
∇sparseERI_2e4c
∇ERI_2e3c(::BasisSet, ::BasisSet, ::Any)
∇ERI_2e2c(::BasisSet, ::Any)
```

### Shell-pair/shell-quartet-level primitives

For callers building an integral-direct gradient or CPHF loop that never
wants the full dense array materialized, single-shell-pair (or
shell-quartet) forms are also available:

```@docs
∇overlap(::BasisSet, ::Int, ::Int, ::Int)
∇kinetic(::BasisSet, ::Int, ::Int, ::Int)
∇nuclear(::BasisSet, ::Int, ::Int, ::Int)
∇ERI_2e4c(::BasisSet, ::Int, ::Int, ::Int, ::Int, ::Int)
∇ERI_2e4c!(::Any, ::BasisSet, ::NTuple{4,Bool}, ::Int, ::Int, ::Int, ::Int, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int32})
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

julia> gS = ∇overlap(bset, 2);  # ∂S/∂R for the first H atom

julia> size(gS)
(7, 7, 3)

julia> gS[2, 6, :]   # ⟨2s_O| shell | 1s_H(2)⟩ block, (x, y, z) components
3-element Vector{Float64}:
 -0.05625663685821103
 -0.0439525017729466
  0.0

julia> gERI = ∇ERI_2e4c(bset, 2);

julia> size(gERI)
(7, 7, 7, 7, 3)
```

For repeated calls, the mutating `∇overlap!`, `∇kinetic!`, `∇nuclear!`,
`∇ERI_2e4c!`, `∇ERI_2e3c!`, and `∇ERI_2e2c!` forms write into a
preallocated output array instead of reallocating.
