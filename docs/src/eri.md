```@meta
CurrentModule = GaussianBasis
```

# Two-Electron Integrals

## Theory

The two-electron repulsion integral (ERI) between four AOs, in chemist's
notation, is
```math
(\mu\nu|\lambda\sigma) = \iint \chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1)\,
\frac{1}{|\mathbf{r}_1-\mathbf{r}_2|}\, \chi_\lambda(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2)\, d\mathbf{r}_1 d\mathbf{r}_2
```
This dense 4-center integral obeys an 8-fold permutational symmetry,
``(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda) =
(\lambda\sigma|\mu\nu) = \ldots``, and its full tensor scales as
``\mathcal{O}(n_\text{bas}^4)`` in memory -- prohibitive for anything but
small systems.

Two related, cheaper integrals underlie **density fitting** (also called
resolution-of-the-identity), which approximates the 4-center ERI by
expanding each electron's charge density in an auxiliary basis:
```math
(P|Q) = \iint \varphi_P(\mathbf{r}_1) \frac{1}{|\mathbf{r}_1-\mathbf{r}_2|} \varphi_Q(\mathbf{r}_2)\, d\mathbf{r}_1 d\mathbf{r}_2
\qquad
(\mu\nu|P) = \iint \chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1) \frac{1}{|\mathbf{r}_1-\mathbf{r}_2|} \varphi_P(\mathbf{r}_2)\, d\mathbf{r}_1 d\mathbf{r}_2
```
where ``\varphi_P, \varphi_Q`` are auxiliary/fitting basis functions. `(P|Q)`
is the 2-center Coulomb metric and `(\mu\nu|P)` the 3-center integral; both
scale far more modestly (``\mathcal{O}(n_\text{aux}^2)`` and
``\mathcal{O}(n_\text{bas}^2 n_\text{aux})`` respectively) than the full
4-center tensor.

For large basis sets, `sparseERI_2e4c` provides a middle ground for the
4-center integral itself: Cauchy-Schwarz screening discards shell quartets
below a magnitude cutoff up front, and only the permutationally-unique
surviving elements are stored, rather than the full dense tensor.

## Usage

```@docs
ERI_2e4c(::BasisSet)
ERI_2e4c(::BasisSet, ::Any, ::Any, ::Any, ::Any)
sparseERI_2e4c
ERI_2e3c(::BasisSet, ::BasisSet)
ERI_2e2c(::BasisSet)
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

julia> I = ERI_2e4c(bset);

julia> size(I)
(7, 7, 7, 7)

julia> I[1, 1, 1, 1]   # (1s_O 1s_O | 1s_O 1s_O)
4.785065998218548

julia> idx, vals = sparseERI_2e4c(bset);

julia> length(idx)     # number of unique, screened-surviving elements
228
```

Density fitting needs an auxiliary `BasisSet` (e.g. a `-jkfit` or `-rifit`
basis) alongside the orbital basis:

```julia-repl
julia> auxbset = BasisSet("cc-pvdz-rifit", water);

julia> Jmetric = ERI_2e2c(auxbset);

julia> size(Jmetric)
(84, 84)

julia> B = ERI_2e3c(bset, auxbset);

julia> size(B)
(7, 7, 84)
```

For repeated calls, `ERI_2e4c!`, `ERI_2e3c!`, and `ERI_2e2c!` write into a
preallocated output array instead of reallocating.
