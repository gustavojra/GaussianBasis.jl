export dipole, dipole!, quadrupole, quadrupole!, octupole, octupole!, hexadecapole, hexadecapole!

# Mutating, shell-pair-level backends for dipole/quadrupole/octupole/
# hexadecapole(BS, i, j): write the raw libcint Cartesian moment integrals
# ⟨i|r|j⟩, ⟨i|rr|j⟩, ... into a caller-supplied `out`. LCint-backend only
# (no ACSint fallback for multipole integrals).
"""
    dipole!(out, BS::BasisSet, i, j)
    dipole!(out, BS::BasisSet)

Mutating counterpart of [`dipole`](@ref): writes into the caller-supplied
`out` instead of allocating. This shell-pair form is the primitive the
full-tensor form builds on.

# Methods

  - `dipole!(out, BS, i, j)`: `out` must be an `(Ni,Nj,3)` array, the block
    for shells `i`/`j` of `BS` (shell indices, not AO indices), the trailing
    axis indexing the `x,y,z` Cartesian components.
  - `dipole!(out, BS)`: `out` must be a dense `nbas × nbas × 3` array.
"""
dipole!(out, BS::BasisSet{LCint}, i, j) = cint1e_r_sph!(out, [i,j], BS.lib)

"""
    quadrupole!(out, BS::BasisSet, i, j)
    quadrupole!(out, BS::BasisSet)

Mutating counterpart of [`quadrupole`](@ref): writes into the caller-supplied
`out` instead of allocating. This shell-pair form is the primitive the
full-tensor form builds on.

# Methods

  - `quadrupole!(out, BS, i, j)`: `out` must be an `(Ni,Nj,3,3)` array, the
    block for shells `i`/`j` of `BS` (shell indices, not AO indices).
  - `quadrupole!(out, BS)`: `out` must be a dense `nbas × nbas × 3 × 3` array.
"""
quadrupole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rr_sph!(out, [i,j], BS.lib)

"""
    octupole!(out, BS::BasisSet, i, j)
    octupole!(out, BS::BasisSet)

Mutating counterpart of [`octupole`](@ref): writes into the caller-supplied
`out` instead of allocating. This shell-pair form is the primitive the
full-tensor form builds on.

# Methods

  - `octupole!(out, BS, i, j)`: `out` must be an `(Ni,Nj,3,3,3)` array, the
    block for shells `i`/`j` of `BS` (shell indices, not AO indices).
  - `octupole!(out, BS)`: `out` must be a dense `nbas × nbas × 3 × 3 × 3`
    array.
"""
octupole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rrr_sph!(out, [i,j], BS.lib)

"""
    hexadecapole!(out, BS::BasisSet, i, j)
    hexadecapole!(out, BS::BasisSet)

Mutating counterpart of [`hexadecapole`](@ref): writes into the
caller-supplied `out` instead of allocating. This shell-pair form is the
primitive the full-tensor form builds on.

# Methods

  - `hexadecapole!(out, BS, i, j)`: `out` must be an `(Ni,Nj,3,3,3,3)`
    array, the block for shells `i`/`j` of `BS` (shell indices, not AO
    indices).
  - `hexadecapole!(out, BS)`: `out` must be a dense
    `nbas × nbas × 3 × 3 × 3 × 3` array.
"""
hexadecapole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rrrr_sph!(out, [i,j], BS.lib)

function dipole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.shells[i]), num_basis(BS.shells[j]), 3)
    dipole!(out, BS, i, j)
    return out
end

function quadrupole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.shells[i]), num_basis(BS.shells[j]), 3, 3)
    quadrupole!(out, BS, i, j)
    return out
end

function octupole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.shells[i]), num_basis(BS.shells[j]), 3, 3, 3)
    octupole!(out, BS, i, j)
    return out
end

function hexadecapole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.shells[i]), num_basis(BS.shells[j]), 3, 3, 3, 3)
    hexadecapole!(out, BS, i, j)
    return out
end

"""
    dipole(BS::BasisSet) -> Array{Float64,3}
    dipole(BS::BasisSet, i, j) -> Array{Float64,3}

Compute the AO electric dipole integral tensor, in atomic units (bohr)
about the origin. Combine with a density matrix and nuclear charges to get
a molecular dipole moment.

# Methods

  - `dipole(BS)`: full tensor for `BS`. Returns a dense `nbas × nbas × 3`
    array, the trailing axis indexing the `x,y,z` Cartesian components.
  - `dipole(BS, i, j)`: just the block for shells `i` and `j` of `BS` (shell
    indices, not AO indices). Returns an `(Ni,Nj,3)` array.

For repeated calls, see `dipole!`, which writes into a preallocated array
instead of allocating.
"""
dipole(BS::BasisSet) = get_1e_matrix(dipole!, BS, 1)

"""
    quadrupole(BS::BasisSet) -> Array{Float64,4}
    quadrupole(BS::BasisSet, i, j) -> Array{Float64,4}

Compute the AO electric quadrupole integral tensor, in atomic units about
the origin; see [`dipole`](@ref) for the general convention.

# Methods

  - `quadrupole(BS)`: full tensor for `BS`. Returns a dense
    `nbas × nbas × 3 × 3` array.
  - `quadrupole(BS, i, j)`: just the block for shells `i` and `j` of `BS`
    (shell indices, not AO indices). Returns an `(Ni,Nj,3,3)` array.

For repeated calls, see `quadrupole!`, which writes into a preallocated
array instead of allocating.
"""
quadrupole(BS::BasisSet) = get_1e_matrix(quadrupole!, BS, 2)

"""
    octupole(BS::BasisSet) -> Array{Float64,5}
    octupole(BS::BasisSet, i, j) -> Array{Float64,5}

Compute the AO electric octupole integral tensor, in atomic units about the
origin; see [`dipole`](@ref) for the general convention.

# Methods

  - `octupole(BS)`: full tensor for `BS`. Returns a dense
    `nbas × nbas × 3 × 3 × 3` array.
  - `octupole(BS, i, j)`: just the block for shells `i` and `j` of `BS`
    (shell indices, not AO indices). Returns an `(Ni,Nj,3,3,3)` array.

For repeated calls, see `octupole!`, which writes into a preallocated array
instead of allocating.
"""
octupole(BS::BasisSet) = get_1e_matrix(octupole!, BS, 3)

"""
    hexadecapole(BS::BasisSet) -> Array{Float64,6}
    hexadecapole(BS::BasisSet, i, j) -> Array{Float64,6}

Compute the AO electric hexadecapole integral tensor, in atomic units about
the origin; see [`dipole`](@ref) for the general convention.

# Methods

  - `hexadecapole(BS)`: full tensor for `BS`. Returns a dense
    `nbas × nbas × 3 × 3 × 3 × 3` array.
  - `hexadecapole(BS, i, j)`: just the block for shells `i` and `j` of `BS`
    (shell indices, not AO indices). Returns an `(Ni,Nj,3,3,3,3)` array.

For repeated calls, see `hexadecapole!`, which writes into a preallocated
array instead of allocating.
"""
hexadecapole(BS::BasisSet) = get_1e_matrix(hexadecapole!, BS, 4)

dipole!(out, BS::BasisSet) = get_1e_matrix!(dipole!, out, BS, 1)
quadrupole!(out, BS::BasisSet) = get_1e_matrix!(quadrupole!, out, BS, 2)
octupole!(out, BS::BasisSet) = get_1e_matrix!(octupole!, out, BS, 3)
hexadecapole!(out, BS::BasisSet) = get_1e_matrix!(hexadecapole!, out, BS, 4)
