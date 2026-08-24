```@meta
CurrentModule = GaussianBasis
```

# Hessians

## Theory

Analytic nuclear Hessians of an integral are its second derivative with
respect to the Cartesian coordinates of two atoms `A`, `B` (`A` and `B` may
be the same atom), e.g.
```math
\frac{\partial^2 S_{\mu\nu}}{\partial \mathbf{R}_A \partial \mathbf{R}_B}, \qquad
\frac{\partial^2 (\mu\nu|\lambda\sigma)}{\partial \mathbf{R}_A \partial \mathbf{R}_B}
```
returned as two extra trailing length-3 axes (``x,y,z`` for `A`, then `B`).
These feed analytic vibrational frequency calculations and second-order
response (CPHF/CPKS) machinery.

!!! warning "Units: `R_A`/`R_B` are in bohr, not Angstrom"
    As with [Gradients](@ref), a [`BasisSet`](@ref) stores and accepts
    nuclear coordinates in Angstrom, but every Hessian here is with respect
    to bohr displacement of each nucleus. So `∇2overlap(bset, iA, iB)` gives
    `∂²S/∂R_iA∂R_iB` per bohr², *not* per Angstrom². To convert to a
    per-Angstrom² derivative, divide by `Molecules.bohr_to_angstrom^2`
    (≈0.529177 Å/bohr, squared).

For nuclear attraction specifically, the second derivative has a shell
piece (both derivatives on shell centers) and a nuclear-charge piece (one
or both derivatives on a moving nuclear charge); `∇2nuclear` handles both
transparently, but the underlying implementation isolates them via a
repositionable point-charge kernel per nucleus -- see
`src/Hessians/NuclearHess.jl` if you need that level of detail.

## Usage

```@docs
∇2overlap(::BasisSet, ::Any, ::Any)
∇2kinetic(::BasisSet, ::Any, ::Any)
∇2nuclear(::BasisSet, ::Any, ::Any)
∇2ERI_2e4c(::BasisSet, ::Int, ::Int, ::Int, ::Int, ::Int, ::Int)
∇2ERI_2e4c!(::Any, ::BasisSet, ::NTuple{4,Bool}, ::NTuple{4,Bool}, ::Int, ::Int, ::Int, ::Int, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int32})
∇2ERI_2e3c(::BasisSet, ::BasisSet, ::Any, ::Any)
∇2ERI_2e2c(::BasisSet, ::Any, ::Any)
```

Shell-quartet-level primitives (`∇2ERI_2e4c(BS,iA,iB,i,j,k,l)`) are also
available for callers assembling an integral-direct Hessian without
materializing the full dense array -- see `?∇2ERI_2e4c` for details.

## Example

```julia-repl
julia> using GaussianBasis, Molecules

julia> water = Molecules.parse_string("""
       O        0.000000000000     -0.143225816552      0.000000000000
       H        1.638036840407      1.136548822547     -0.000000000000
       H       -1.638036840407      1.136548822547     -0.000000000000
       """);

julia> bset = BasisSet("sto-3g", water);

julia> hS = ∇2overlap(bset, 1, 2);  # ∂²S/∂R_O∂R_H(first)

julia> size(hS)
(7, 7, 3, 3)

julia> hV = ∇2nuclear(bset, 1, 1);  # diagonal block (A = B = O)

julia> size(hV)
(7, 7, 3, 3)
```

For repeated calls, the mutating `∇2overlap!`, `∇2kinetic!`, `∇2nuclear!`,
`∇2ERI_2e4c!`, `∇2ERI_2e3c!`, and `∇2ERI_2e2c!` forms write into a
preallocated output array instead of reallocating.
