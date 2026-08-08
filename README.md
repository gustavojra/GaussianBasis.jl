<p align="center">
  <img src="assets/gblogo.png" width="600" alt=""/>
</p>

<table align="center">
  <tr>
    <th>CI</th>
    <th>Coverage</th>
    <th>License</th>
  </tr>
  <tr>
    <td align="center">
      <a href=https://github.com/FermiQC/GaussianBasis.jl/actions/workflows/CI.yml>
      <img src=https://github.com/FermiQC/GaussianBasis.jl/actions/workflows/CI.yml/badge.svg>
      </a> 
    </td>
    <td align="center">
      <a href=https://codecov.io/gh/FermiQC/GaussianBasis.jl>
      <img src=https://codecov.io/gh/FermiQC/GaussianBasis.jl/branch/main/graph/badge.svg?token=JNouJPwoHm>
      </a> 
    </td>
    <td align="center">
      <a href=https://github.com/FermiQC/GaussianBasis.jl/blob/main/LICENSE>
      <img src=https://img.shields.io/badge/License-MIT-blue.svg>
      </a>
    </td>
  </tr>
</table>

GaussianBasis offers high-level utilities for molecular integral computations.

Current features include:

- Basis set parsing (`gbs` format)
- Standard basis set files from [BSE](https://www.basissetexchange.org/)
- One-electron integral (1e)
- Two-electron two-center integral (2e2c)
- Two-electrons three-center integral (2e3c)
- Two-electrons four-center integral (2e4c)
- Analytic gradients (first derivatives w.r.t. nuclear coordinates) for all
  of the above
- Analytic Hessians (second derivatives) for one-electron integrals
  (overlap/kinetic/nuclear attraction) and the 2-center/3-center two-electron
  integrals (density fitting). The dense 4-center ERI Hessian is not yet
  available in GaussianBasis.jl itself -- see
  [Derivatives](#derivatives-gradients-and-hessians) below.

Integral computations use by default the integral library [libcint](https://github.com/sunqm/libcint) *via* [libcint_jll.jl](https://github.com/JuliaBinaryWrappers/libcint_jll.jl). A simple Julia-written integral module `Acsint.jl` is also available, but it is significantly slower than the `libcint`.  

# Basic Usage

The simplest way to use the code is by first creating a `BasisSet` object. For example
```julia
julia> bset = BasisSet("sto-3g", """
              H        0.00      0.00     0.00                 
              H        0.76      0.00     0.00""")
sto-3g Basis Set
Type: Spherical   Backend: Libcint
Number of shells: 2
Number of basis:  2

H: 1s 
H: 1s
```
Next, call the desired integral function with the `BasisSet` object as the argument. Let's take the `overlap` function as an example:
```julia
julia> overlap(bset)
2×2 Matrix{Float64}:
 1.0       0.646804
 0.646804  1.0
```

| Function      | Description | Formula |
|---------------|-------------|:-------:|
| `overlap`       | Overlap between two basis functions | ![S](assets/ovlp.png)|
| `kinetic`       | Kinetic integral | ![T](assets/kin.png)|
| `nuclear`       | Nuclear attraction integral  | ![V](assets/nuc.png)|
| `ERI_2e4c`       | Electron repulsion integral - returns a full rank-4 tensor! | ![ERI](assets/4cERI.png)|
| `sparseERI_2e4c`       | Electron repulsion integral - returns non-zero elements along with a index tuple | ![sERI](assets/4cERI.png)|
| `ERI_2e3c`       | Electron repulsion integral over three centers. **Note:** this function requires another basis set as the second argument (that is the auxiliary basis set in [Density Fitting](http://vergil.chemistry.gatech.edu/notes/df.pdf)). It must be called as `ERI_2c3c(bset, aux)` | ![3cERI](assets/3cERI.png)|
| `ERI_2e2c`       | Electron repulsion integral over two centers  | ![2cERI](assets/2cERI.png)|

# Derivatives (Gradients and Hessians)

Analytic derivatives w.r.t. nuclear Cartesian coordinates are available for
every integral above. Naming follows the integral it differentiates, with a
`∇` prefix for the first derivative (gradient) and `∇2` for the second
(Hessian): `overlap` -> `∇overlap` -> `∇2overlap`, and so on. The atom being
differentiated is given as an integer index `iA` into `bset.atoms` (Hessians
take two, `iA, iB`, one per derivative order); the extra trailing axes on the
output hold the 3 Cartesian components per derivative order (`(..., 3)` for
a gradient, `(..., 3, 3)` for a Hessian). As with the plain integrals, every
function has an in-place `!` form and functions that take two basis sets
(`ERI_2e3c(bset, auxbset)`-style) keep that same two-basis-set calling
convention at every derivative order.

## File organization

Mirrors the integral files one directory over, by derivative order:

| Order | Bundle file | Sub-files |
|---|---|---|
| 0 (integrals) | `Integrals.jl` | `Integrals/OneElectron.jl`, `TwoElectronTwoCenter.jl`, `TwoElectronThreeCenter.jl`, `TwoElectronFourCenter.jl`, `Multipole.jl` |
| 1 (gradients) | `Gradients.jl` | `Gradients/OneElectronGrad.jl`, `TwoElectronGrad.jl`, `FiniteDifferences.jl` |
| 2 (Hessians) | `Hessians.jl` | `Hessians/OneElectronHess.jl`, `NuclearHess.jl`, `FiniteDifferences.jl` |

Two naming departures worth knowing about if you're looking for a function:

- Two-electron integrals are split by center-count at the integral level
  (`TwoElectronTwoCenter.jl`/`ThreeCenter.jl`/`FourCenter.jl`) but bundled
  into a single `TwoElectronGrad.jl` at the gradient level (4-center,
  3-center, and 2-center gradients all live there together).
- Nuclear attraction gets its own file only at the Hessian level
  (`NuclearHess.jl`, separate from `OneElectronHess.jl`) -- see below for
  why.

Each `FiniteDifferences.jl` holds central-difference reference
implementations (`∇FD_*`/`∇2FD_*`) used to validate the analytic code one
derivative order down (gradients are checked against finite differences of
the integrals; Hessians against finite differences of the gradients), not
meant for production use.

## Gradients

| Function | Output shape | Description |
|---|---|---|
| `∇overlap(bset, iA)` | `(nbas,nbas,3)` | Overlap gradient |
| `∇kinetic(bset, iA)` | `(nbas,nbas,3)` | Kinetic energy gradient |
| `∇nuclear(bset, iA)` | `(nbas,nbas,3)` | Nuclear attraction gradient |
| `∇ERI_2e4c(bset, iA)` | `(nbas,nbas,nbas,nbas,3)` | Dense 4-center ERI gradient |
| `∇sparseERI_2e4c(bset, iA)` | `(idx, ∇x, ∇y, ∇z)` | Screened, permutation-compressed 4-center ERI gradient -- see below |
| `∇ERI_2e3c(bset, auxbset, iA)` | `(nbas,nbas,naux,3)` | 3-center ERI gradient (density fitting) |
| `∇ERI_2e2c(auxbset, iA)` | `(naux,naux,3)` | 2-center (auxiliary metric) ERI gradient (density fitting) |

## Hessians

| Function | Output shape | Description |
|---|---|---|
| `∇2overlap(bset, iA, iB)` | `(nbas,nbas,3,3)` | Overlap Hessian |
| `∇2kinetic(bset, iA, iB)` | `(nbas,nbas,3,3)` | Kinetic energy Hessian |
| `∇2nuclear(bset, iA, iB)` | `(nbas,nbas,3,3)` | Nuclear attraction Hessian |
| `∇2ERI_2e2c(auxbset, iA, iB)` | `(naux,naux,3,3)` | 2-center (auxiliary metric) ERI Hessian (density fitting) |
| `∇2ERI_2e3c(bset, auxbset, iA, iB)` | `(nbas,nbas,naux,3,3)` | 3-center ERI Hessian (density fitting) |

The dense 4-center ERI Hessian doesn't exist in GaussianBasis.jl -- it's
currently implemented ad hoc inside Fermi.jl (calling raw libcint kernels
directly rather than going through a GaussianBasis.jl wrapper).

## Where Hessian-level functions are *not* just "one more derivative"

It's tempting to assume every `∇2X` is a mechanical extension of the
matching `∇X`, differentiated once more the same way. Three cases break that
assumption, and are worth understanding before extending this pattern
further (e.g. to the dense 4-center ERI Hessian):

**`∇sparseERI_2e4c` isn't a compressed `∇ERI_2e4c`, it's a different
algorithm.** `sparseERI_2e4c` (the *energy* integral) applies Cauchy-Schwarz
shell-pair screening -- skipping whole shell quartets whose contribution is
provably negligible -- *before* computing anything, which is where its
performance advantage over the dense `ERI_2e4c` actually comes from.
`∇sparseERI_2e4c` reuses that same screening bound (`σ_ij := sqrt(max|(ij|
ij)|)`, computed from the plain, undifferentiated integrals -- see
`schwarz_bounds(bset)`, which callers making repeated `∇sparseERI_2e4c`
calls across atoms should compute once and pass in via the `ij_vals`/
`σvals` keyword arguments, rather than recomputing it, atom-independent, on
every call). The screening bound is a valid proxy for the *derivative*
integral too: differentiating a Gaussian-product ERI only introduces a
bounded polynomial prefactor via the chain rule, so the exponential
shell-pair falloff that makes the bound work for the energy integral is
unchanged for its derivative. Symmetry-wise, `∇sparseERI_2e4c` also drops
whole quartets `∇ERI_2e4c` cannot: any quartet with all four indices on the
same atom, or none of them on it, has an *exactly* zero derivative by
translational invariance, not just a small one -- correctness, not an
approximation.

**The nuclear attraction Hessian needs different kernels than
overlap/kinetic's, not just one more derivative of the same ones.** Overlap
and kinetic depend on exactly two positions (the two shell centers), so
their Hessian is a direct atom-pair generalization of the gradient's atom-
membership logic -- same kernel family (`ipip`/cross), one more derivative
axis. Nuclear attraction depends on a *third* position: whichever nucleus
supplies the `-Z/|r-R|` potential. A derivative w.r.t. atom A can only land
on that potential role when the supplying nucleus *is* A, so the second-
derivative kernels have to isolate one nucleus at a time (via a
repositionable `rinv` operator) and combine two genuinely different pieces
-- "both derivatives on a shell center" (same kernel family as
overlap/kinetic, nucleus-independent) and "at least one derivative on the
nuclear-charge position" (a different kernel family, `ipiprinv`/`iprinvip`,
applied once per relevant nucleus) -- without double-counting. See
`Hessians/NuclearHess.jl`'s header comment for the full derivation; this is
also why nuclear attraction is split into its own file at the Hessian level
but not at the gradient level.

**`∇2ERI_2e3c` has 6 shell-position placements, not the 2-shell case's
2.** Overlap/kinetic and the metric `∇2ERI_2e2c` only ever have two
positions (same-shell, cross-shell). The 3-center integral `(μν|P)` has
three unpaired positions -- μ, ν (regular, symmetric under swap), and P
(auxiliary) -- giving 3 same-shell placements (μμ, νν, PP) and 3 *distinct*
cross placements (μν, μP, νP), each needing its own libcint kernel or
kernel-plus-permutation (`cint3c2e_ipip1`/`ipip2` for same-shell,
`ipvip1`/`ip1ip2` for cross -- see `Hessians/TwoElectronThreeCenterHess.jl`'s
header for the full mapping, mirroring how Fermi.jl's own 4-center ERI
Hessian already worked out the same `ipvip1`/`ip1ip2` naming pattern one
shell-count up).

# Advanced Usage

## Basis Functions
`BasisFunction` object is the central data type within this package. Here, `BasisFunction` is an abstract type with two concrete structures: `SphericalShell` and `CartesianShell`. By default `SphericalShell` is created. In general a spherical basis function is

![BF](assets/bf.png)

where the sum goes over primitive functions. A `BasisFunction` object contains the data to reproduce the mathematical object, i.e. the angular momentum number (***l***), expansion coefficients (***c<sub>n</sub>***), and exponential factors (***&xi;<sub>n</sub>***). We can create a basis function by passing these arguments orderly:
```julia
julia> using StaticArrays
julia> atom = GaussianBasis.Atom(8, 16.0, [1.0, 0.0, 0.0])
julia> bf = BasisFunction(1, SVector(1/√2, 1/√2), SVector(5.0, 1.2), atom)
P shell with 3 basis built from 2 primitive gaussians

χ₁₋₁ =    0.7071067812⋅Y₁₋₁⋅r¹⋅exp(-5.0⋅r²)
     +    0.7071067812⋅Y₁₋₁⋅r¹⋅exp(-1.2⋅r²)

χ₁₀  =    0.7071067812⋅Y₁₀⋅r¹⋅exp(-5.0⋅r²)
     +    0.7071067812⋅Y₁₀⋅r¹⋅exp(-1.2⋅r²)

χ₁₁  =    0.7071067812⋅Y₁₁⋅r¹⋅exp(-5.0⋅r²)
     +    0.7071067812⋅Y₁₁⋅r¹⋅exp(-1.2⋅r²)
```
We can now check the fields (attributes):
```julia
julia> bf.l
1

julia> bf.coef
2-element SVector{2, Float64} with indices SOneTo(2):
 0.7071067811865475
 0.7071067811865475

julia> bf.exp
2-element SVector{2, Float64} with indices SOneTo(2):
 5.0
 1.2
 ```
 Note that `exp` and `coef` are expected to be `SVector` from `StaticArrays`. 

 ## Basis Set

 The `BasisSet` object is the main ingredient for integrals. It can be created in a number of ways:

 - The highest level approach takes two strings as arguments, one for the basis set name and another for the XYZ file. See *Basic Usage*.

 - You can pass your vector of `Atom` structures instead of an XYZ string as the second argument. `GaussianBasis` uses the `Atom` structure from [Molecules.jl](https://github.com/FermiQC/GaussianBasis.jl).
  ```julia
atoms = GaussianBasis.parse_string("""
              H        0.00      0.00     0.00                 
              H        0.76      0.00     0.00""")
BasisSet("sto-3g", atoms)
```

 - Finally, instead of searching into `GaussianBasis/lib` for a basis set file matching the desired name, you can construct your own from scratch. We further discuss this approach below. 

 Basis sets are mainly composed of two arrays: a vector of atoms and a vector of basis functions objects. We can construct both manually for maximum flexibility: 
 ```julia
julia> h2 = GaussianBasis.parse_string(
   "H 0.0 0.0 0.0
    H 0.0 0.0 0.7"
)
2-element Vector{Atom{Int16, Float64}}:
 Atom{Int16, Float64}(1, 1.008, [0.0, 0.0, 0.0])
 Atom{Int16, Float64}(1, 1.008, [0.0, 0.0, 0.7])
 ```
Next, we create a vector of basis functions.
```julia
julia> shells = [BasisFunction(0, SVector(0.5215367271), SVector(0.122), h2[1]),
BasisFunction(0, SVector(0.5215367271), SVector(0.122), h2[2]),
BasisFunction(1, SVector(1.9584045349), SVector(0.727), h2[2])];
```
Finally, we create the basis set object. Note that, you got to make sure your procedure is consistent. The atoms used to construct the basis set object must be in the `atom` vector, otherwise unexpected results may arise. 
```julia
julia> bset = BasisSet("UnequalHydrogens", h2, shells)
UnequalHydrogens Basis Set
Type: Spherical{Molecules.Atom, 1, Float64}   Backend: Libcint
Number of shells: 3
Number of basis:  5

H: 1s 
H: 1s 1p
```
The most import fields here are:
```julia
julia> bset.name == "UnequalHydrogens"
true
julia> bset.basis == shells 
true
julia> bset.atoms == h2
true
```

### Integrals over different basis sets

Functions such as `ERI_2e3c` require two basis set as arguments. Looking at the corresponding equation
![3cERI](assets/3cERI.png) we see two basis set: ***&Chi;*** and ***P***. If your first basis set has 2 basis functions and the second has 4, your output array is a 2x2x4 tensor. For example
```julia
julia> b1 = BasisSet("sto-3g", """
              H        0.00      0.00     0.00                 
              H        0.76      0.00     0.00""")
julia> b2 = BasisSet("3-21g", """
              H        0.00      0.00     0.00                 
              H        0.76      0.00     0.00""")
julia> ERI_2e3c(b1,b2)
2×2×4 Array{Float64, 3}:
[:, :, 1] =
 3.26737  1.85666
 1.85666  2.44615

[:, :, 2] =
 6.18932  3.83049
 3.83049  5.60161

[:, :, 3] =
 2.44615  1.85666
 1.85666  3.26737

[:, :, 4] =
 5.60161  3.83049
 3.83049  6.18932
 ```
One electron integrals can also be employed with different basis set. 
```julia
julia> overlap(b1, b2)
2×4 Matrix{Float64}:
 0.914077  0.899458  0.473201  0.708339
 0.473201  0.708339  0.914077  0.899458

julia> kinetic(b1, b2)
2×4 Matrix{Float64}:
 1.03401  0.314867  0.20091  0.203163
 0.20091  0.203163  1.03401  0.314867
```
This can be useful when working with projections from one basis set onto another. 

### Computing integrals element-wise

For all integrals, you can get the full array by using the general syntax `integral(basisset)` (e.g. `overlap(bset)` or `ERI_2e4c(bset)`). Alternatively, you can specify a shell combination for which the integral must be computed
```julia
julia> ERI_2e4c(b1, 1,2,2,1)
1×1×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.2845189435761272

julia> kinetic(b1, 1,2)
1×1 Matrix{Float64}:
 0.2252049038643092
 ```
Mutating versions of the functions are also available 
```julia
julia> S = zeros(2,2);
julia> overlap!(S, b1)
julia> S
2×2 Matrix{Float64}:
 1.0       0.646804
 0.646804  1.0
 ```

### Evaluating orbital amplitudes

The function `atomic_orbital_amplitude(basisset, i, r)` can be used to calculate the atomic orbital amplitude of the `i`th basis function at positions `r`. `r` can be either a 3-vector for a single position or a 3×… array for calculating many positions efficiently at once.

```julia
julia> bset = BasisSet("sto-3g", "H 0 0 0");
julia> atomic_orbital_amplitude(bset, 1, [0.0,0.0,0.0])
0.6282468778403579
julia> atomic_orbital_amplitude(bset, 1, [0.1;0.2;0.3;;-0.1;0.3;-0.2])
2-element Vector{Float64}:
 0.49840190793869554
 0.49840190793869554
```
