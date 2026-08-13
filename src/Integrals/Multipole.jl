export dipole, quadrupole

# Mutating, shell-pair-level backends for dipole/quadrupole/octupole/
# hexadecapole(BS, i, j): write the raw libcint Cartesian moment integrals
# ⟨i|r|j⟩, ⟨i|rr|j⟩, ... into a caller-supplied `out`. LCint-backend only
# (no ACSint fallback for multipole integrals).
dipole!(out, BS::BasisSet{LCint}, i, j) = cint1e_r_sph!(out, [i,j], BS.lib)
quadrupole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rr_sph!(out, [i,j], BS.lib)
octupole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rrr_sph!(out, [i,j], BS.lib)
hexadecapole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rrrr_sph!(out, [i,j], BS.lib)

"""
    dipole(BS::BasisSet, i, j) -> Array{Float64,3}

Compute the AO electric dipole integral block `⟨i|r|j⟩` for shells `i,j` of
`BS`, in atomic units (bohr) about the origin. Returned as an `(Ni,Nj,3)`
array, the trailing axis indexing the `x,y,z` Cartesian components. For the
full dipole tensor, see [`dipole(BS)`](@ref dipole(::BasisSet)).
"""
function dipole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]), 3)
    dipole!(out, BS, i, j)
    return out
end

"""
    quadrupole(BS::BasisSet, i, j) -> Array{Float64,4}

Compute the AO electric quadrupole integral block `⟨i|rr|j⟩` for shells
`i,j` of `BS`, in atomic units about the origin. Returned as an
`(Ni,Nj,3,3)` array, the trailing two axes indexing the Cartesian
components of each `r` factor. For the full tensor, see
[`quadrupole(BS)`](@ref quadrupole(::BasisSet)).
"""
function quadrupole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]), 3, 3)
    quadrupole!(out, BS, i, j)
    return out
end

"""
    octupole(BS::BasisSet, i, j) -> Array{Float64,5}

Compute the AO electric octupole integral block `⟨i|rrr|j⟩` for shells
`i,j` of `BS`, in atomic units about the origin. Returned as an
`(Ni,Nj,3,3,3)` array, the trailing three axes indexing the Cartesian
components of each `r` factor. For the full tensor, see `octupole(BS)`.
"""
function octupole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]), 3, 3, 3)
    octupole!(out, BS, i, j)
    return out
end

"""
    hexadecapole(BS::BasisSet, i, j) -> Array{Float64,6}

Compute the AO electric hexadecapole integral block `⟨i|rrrr|j⟩` for shells
`i,j` of `BS`, in atomic units about the origin. Returned as an
`(Ni,Nj,3,3,3,3)` array, the trailing four axes indexing the Cartesian
components of each `r` factor. For the full tensor, see `hexadecapole(BS)`.
"""
function hexadecapole(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]), 3, 3, 3, 3)
    hexadecapole!(out, BS, i, j)
    return out
end

"""
    dipole(BS::BasisSet) -> Array{Float64,3}

Compute the full AO electric dipole integral tensor for `BS`, in atomic
units (bohr) about the origin. Returns a dense `nbas × nbas × 3` array, the
trailing axis indexing the `x,y,z` Cartesian components. Combine with a
density matrix and nuclear charges to get a molecular dipole moment.
"""
dipole(BS::BasisSet) = get_1e_matrix(dipole!, BS, 1)

"""
    quadrupole(BS::BasisSet) -> Array{Float64,4}

Full AO electric quadrupole integral tensor for `BS`, in atomic units about
the origin. Returns a dense `nbas × nbas × 3 × 3` array; see
[`dipole(BS)`](@ref dipole(::BasisSet)) for the general convention.
"""
quadrupole(BS::BasisSet) = get_1e_matrix(quadrupole!, BS, 2)

"""
    octupole(BS::BasisSet) -> Array{Float64,5}

Full AO electric octupole integral tensor for `BS`, in atomic units about
the origin. Returns a dense `nbas × nbas × 3 × 3 × 3` array; see
[`dipole(BS)`](@ref dipole(::BasisSet)) for the general convention.
"""
octupole(BS::BasisSet) = get_1e_matrix(octupole!, BS, 3)

"""
    hexadecapole(BS::BasisSet) -> Array{Float64,6}

Full AO electric hexadecapole integral tensor for `BS`, in atomic units
about the origin. Returns a dense `nbas × nbas × 3 × 3 × 3 × 3` array; see
[`dipole(BS)`](@ref dipole(::BasisSet)) for the general convention.
"""
hexadecapole(BS::BasisSet) = get_1e_matrix(hexadecapole!, BS, 4)
