module GaussianBasis

using Molecules
using Format
using StaticArrays
import Molecules: Atom, symbol, parse_file, parse_string

export ShellFunction, BasisSet, SphericalShell, CartesianShell, get_shell, ACSint, LCint, atomic_orbital_amplitude

abstract type ShellFunction end

@doc raw"""
     SphericalShell

Object representing a shell of basis functions in Spherical coordinates. 
The shells types are s, p, d, ... corresponding to angular momentum numbers l = 0, 1, 2, ...
For each shell, there are 2l+1 basis functions corresponding to the magnetic quantum number m = -l, ..., l.

# Fields

| Name        |   Description |
|:------------|:-----------------------------------------------------------|
|`l`          | Angular momentum number |
|`coef`       | Array holding expansion coefficients for the primitive functions |
|`exp`        | Array holding exponents for primitive functions          |

# Examples

```julia
julia> atom = Atom(1, 1.008, [0.0, 0.0, 0.0])
julia> p_shell = SphericalShell(1, [1/√2, 1/√2], [5.0, 1.2], atom)
P shell on Hydrogen at position [0.0, 0.0, 0.0] Å
Contains 3 basis functions built from 2 primitive gaussians

χ₁₋₁ =    0.7071067812⋅Y₁₋₁⋅r¹⋅exp(-5.0⋅r²)
     +    0.7071067812⋅Y₁₋₁⋅r¹⋅exp(-1.2⋅r²)

χ₁₀  =    0.7071067812⋅Y₁₀⋅r¹⋅exp(-5.0⋅r²)
     +    0.7071067812⋅Y₁₀⋅r¹⋅exp(-1.2⋅r²)

χ₁₁  =    0.7071067812⋅Y₁₁⋅r¹⋅exp(-5.0⋅r²)
     +    0.7071067812⋅Y₁₁⋅r¹⋅exp(-1.2⋅r²)
```
"""
struct SphericalShell{A<:Atom} <: ShellFunction
    l::Int
    coef::Vector{Float64}
    exp::Vector{Float64}
    atom::A
end

@doc raw"""
     CartesianShell

Object representing a shell of basis functions in Cartesian coordinates. 
The shells types are s, p, d, ... corresponding to angular momentum numbers l = 0, 1, 2, ...
Note that the number of basis functions in a Cartesian shell is given by (l+1)(l+2)/2, which is larger 
than the number of basis functions in a Spherical shell (2l+1) for l > 1.
That is, m_l values are not well defined in Cartesian shells.

# Fields

| Name        |   Description |
|:------------|:-----------------------------------------------------------|
|`l`          | Pseudo-angular momentum number l = a+b+c for xᵃyᵇzᶜ |
|`coef`       | Array holding expansion coefficients for the primitive functions |
|`exp`        | Array holding exponents for primitive functions          |

# Examples

```julia
julia> atom = Atom(1, 1.008, [0.0, 0.0, 0.0])
julia> d_shell = CartesianShell(2, [1/√2], [5.0], atom)
D shell on Hydrogen at position [0.0, 0.0, 0.0] Å
Contains 6 basis functions built from 1 primitive gaussian

χ(x²) =    0.7071067812⋅x²⋅exp(-5.0⋅r²)

χ(xy) =    0.7071067812⋅xy⋅exp(-5.0⋅r²)

χ(xz) =    0.7071067812⋅xz⋅exp(-5.0⋅r²)

χ(y²) =    0.7071067812⋅y²⋅exp(-5.0⋅r²)

χ(yz) =    0.7071067812⋅yz⋅exp(-5.0⋅r²)

χ(z²) =    0.7071067812⋅z²⋅exp(-5.0⋅r²)
```
"""
struct CartesianShell{A<:Atom} <: ShellFunction
    l::Int
    coef::Vector{Float64}
    exp::Vector{Float64}
    atom::A
end

# Shells are created Spherical by default
ShellFunction(l, coef, exp, atom) = SphericalShell(l,coef,exp, atom)

# Deprecated alias: `BasisFunction` was renamed to `ShellFunction` since one
# object of this type is a whole shell (potentially several AOs), not a
# single basis function -- see `BasisSet.nbas` vs. `BasisSet.nshells`.
Base.@deprecate_binding BasisFunction ShellFunction

# Number of basis in a shell
num_basis(B::CartesianShell) = ((B.l + 1) * (B.l + 2)) ÷ 2
num_basis(B::SphericalShell) = 2*B.l + 1

function index2(i, j)
    if i < j
        return (j * (j + 1)) >> 1 + i
    else
        return (i * (i + 1)) >> 1 + j
    end
end

function index4(i,j,k,l)
    return index2(index2(i,j), index2(k,l))
end

include("BasisParser.jl")
include("BasisSet.jl")
include("Misc.jl")
include("Libcint.jl")
include("Acsint.jl")
include("Integrals.jl")
include("Gradients.jl")
include("Hessians.jl")

end # module
