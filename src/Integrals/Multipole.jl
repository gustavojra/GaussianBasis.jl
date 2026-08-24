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
dipole!(out, BS::BasisSet{LCint}, i, j) = cint1e_r_sph!(out, @SVector([i,j]), BS.lib)

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
quadrupole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rr_sph!(out, @SVector([i,j]), BS.lib)

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
octupole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rrr_sph!(out, @SVector([i,j]), BS.lib)

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
hexadecapole!(out, BS::BasisSet{LCint}, i, j) = cint1e_rrrr_sph!(out, @SVector([i,j]), BS.lib)

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
dipole(BS::BasisSet) = get_multipole_matrix(dipole!, BS, Val(1))

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
quadrupole(BS::BasisSet) = get_multipole_matrix(quadrupole!, BS, Val(2))

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
octupole(BS::BasisSet) = get_multipole_matrix(octupole!, BS, Val(3))

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
hexadecapole(BS::BasisSet) = get_multipole_matrix(hexadecapole!, BS, Val(4))

dipole!(out, BS::BasisSet) = get_multipole_matrix!(dipole!, out, BS, Val(1))
quadrupole!(out, BS::BasisSet) = get_multipole_matrix!(quadrupole!, out, BS, Val(2))
octupole!(out, BS::BasisSet) = get_multipole_matrix!(octupole!, out, BS, Val(3))
hexadecapole!(out, BS::BasisSet) = get_multipole_matrix!(hexadecapole!, out, BS, Val(4))

function get_multipole_matrix(callback, BS::BasisSet, ::Val{N}) where N
    out = zeros(eltype(BS.atoms[1].xyz), BS.nbas, BS.nbas, ntuple(_ -> 3, Val(N))...)
    return get_multipole_matrix!(callback, out, BS, Val(N))
end

# Rank>0 (dipole/quadrupole/octupole/hexadecapole) analogue of
# `get_1e_matrix!` (OneElectron.jl): the number of trailing Cartesian axes
# is `N`, carried as a `Val{N}` type parameter rather than a runtime
# `Integer` -- with a runtime rank, `Iterators.product(Iterators.repeated(1:3,
# rank)...)` splats a runtime-length argument list, which Julia can't give a
# concrete return type (`ProductIterator{T} where T<:Tuple{Vararg{UnitRange}}`),
# so the per-component loop was dynamically dispatched regardless of how the
# shell-pair block was copied into `out` afterward. With `N` known at compile
# time, `CartesianIndices(ntuple(_->3, Val(N)))` is fully concrete, and the
# copy itself uses a plain scalar double loop (see `get_1e_matrix!`'s
# docstring for why that beats range-indexed dot-broadcast).
function get_multipole_matrix!(callback, out, BS::BasisSet, ::Val{N}) where N

    fill!(out, 0.0)

    Nvals = num_basis.(BS.shells)
    Nmax = maximum(Nvals)

    ao_offset = cumsum(Nvals) .- Nvals

    ijs = unique_ij(BS.nshells)

    dims = ntuple(_ -> 3, Val(N))
    ncomp = 3^N

    if Threads.nthreads() == 1 || ncomp * BS.nbas^2 < THREADING_THRESHOLD_1E
        # See THREADING_THRESHOLD_1E -- same rationale as get_1e_matrix!, with
        # the 3^N Cartesian components counted into the work estimate.
        buf = zeros(Cdouble, ncomp * Nmax^2)
        for (i,j) in ijs
            _scatter_multipole!(callback, out, BS, i, j, buf, Nvals, ao_offset, dims)
        end
    else
        allocate(body) = body(zeros(Cdouble, ncomp * Nmax^2))
        workerpool(allocate, ijs; chunksize=10) do (i,j), buf
            _scatter_multipole!(callback, out, BS, i, j, buf, Nvals, ao_offset, dims)
        end
    end
    return out
end

# One shell pair for the rank>0 builders: evaluate into `buf`, then scatter each
# of the 3^N Cartesian-component blocks into `out` with its i<->j mirror.
# Shared by the serial and threaded paths so they can't drift apart.
@inline function _scatter_multipole!(callback, out, BS, i, j, buf, Nvals, ao_offset, dims)
    @inbounds begin
        Ni = Nvals[i]
        Nj = Nvals[j]
        Nij = Ni*Nj
        ioff = ao_offset[i]
        joff = ao_offset[j]

        callback(buf, BS, i, j)

        for (n, ks) in enumerate(CartesianIndices(dims))
            base = Nij*(n-1)
            kt = Tuple(ks)
            for js = 1:Nj, is = 1:Ni
                v = buf[base + is + Ni*(js-1)]
                out[ioff+is, joff+js, kt...] = v
                if i != j
                    out[joff+js, ioff+is, kt...] = v
                end
            end
        end
    end
end
