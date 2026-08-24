###########################################################
###########################################################
#                        Overlap                          #
###########################################################
###########################################################

# GaussianBasis-idiomatic shell-pair primitive (hides libcint's shell
# indexing/buffer-layout plumbing the same way `overlap!(out, BS, i, j)`
# does for the plain integrals), derived by symbolically tracing the
# existing, already-validated whole-array gradient loop (matching its
# transpose-mirroring) for an arbitrary shell pair (i,j) in either order,
# rather than re-deriving the underlying integral calculus from scratch --
# lower risk of introducing a NEW error, though this is still checked
# against the whole-array form in the test suite before being trusted.
"""
    ∇overlap_μ!(out, BS::BasisSet{LCint}, i::Int, j::Int)

Libcint call for the overlap gradient block differentiated with respect to
shell `i` (the "μ" AO), for shells `i,j` of `BS`, already sign-flipped to
the nuclear-coordinate convention: libcint's `cint1e_ipovlp_sph!` computes
`∂χ/∂r` (w.r.t. the electron coordinate), but a GTO shell only depends on
`r` and its center `R` through `r-R`, so `∂χ/∂R = -∂χ/∂r` -- this writes
`-cint1e_ipovlp_sph!(...)`, i.e. `∂S/∂R_i` directly, matching every other
`∇`-prefixed function in this module.

> No bounds checking, no zero-block skipping, no output-size validation.
> `i,j` must be valid shell indices (`1:BS.nshells`) and `out` must be
> exactly `(Ni,Nj,3)` -- violating either is undefined behavior at the C
> level, not a catchable Julia error: an out-of-range shell index can
> segfault the whole process, and an undersized `out` can silently write
> past its end into unrelated memory. Prefer `∇overlap!`/`∇overlap` unless
> you're writing your own integral-direct loop and have already validated
> `i,j` and `out` yourself.
"""
function ∇overlap_μ!(out, BS::BasisSet{LCint}, i::Int, j::Int)
    cint1e_ipovlp_sph!(out, [i, j], BS.lib)
    out .*= -1.0
    return out
end

"""
    ∇overlap_ν!(out, BS::BasisSet{LCint}, i::Int, j::Int)

Same as [`∇overlap_μ!`](@ref), differentiated with respect to shell `j`
(the "ν" AO) instead of `i`. Same lack of bounds checking applies.
"""
function ∇overlap_ν!(out, BS::BasisSet{LCint}, i::Int, j::Int)
    cint1e_ipovlp_sph!(out, [j, i], BS.lib)
    out .*= -1.0
    return out
end

"""
    ∇overlap!(out, BS::BasisSet{LCint}, A::Int, i::Int, j::Int)
    ∇overlap!(out, BS::BasisSet, A)

Mutating counterpart of [`∇overlap`](@ref): writes into the caller-supplied
`out` instead of allocating. The shell-pair form is the primitive the
full-tensor form builds on -- a direct libcint call, exactly zero (returned
without any libcint call) whenever `i,j` are BOTH on atom `A` or BOTH off
it: `S_ij` doesn't depend on `R_A` at all if neither shell sits there, and
translating `i,j` together with atom `A` (both belong to it) leaves their
relative separation -- and so `S_ij` -- unchanged.

Only implemented for the `LCint` backend -- there is no `ACSint` fallback
for gradients.

# Methods

  - `∇overlap!(out, BS, A, i, j)`: `out` must be `(Ni,Nj,3)`, the block
    for shells `i,j` of `BS` (shell indices, not AO indices).
  - `∇overlap!(out, BS, A)`: `out` must be a dense `nbas × nbas × 3`
    array.
"""
function ∇overlap!(out, BS::BasisSet{LCint}, A::Int, i::Int, j::Int)

    i_on_A, j_on_A = GaussianBasis.on_atom_flags(BS, A, i, j)

    if i_on_A == j_on_A
        out .= 0.0
        return out
    end

    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])

    if size(out) != (Ni, Nj, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Ni, $Nj, 3)"))
    end

    if i_on_A
        # Derivative lands on shell i (the one on atom A) directly -- raw
        # libcint output for a [i,j] call is already (Ni,Nj,3) in memory,
        # so write straight into `out`, no intermediate buffer needed.
        ∇overlap_μ!(out, BS, i, j)
    else
        # Derivative lands on shell j (the one on atom A); compute via [j,i]
        # (j as the differentiated, first, shell) -- raw libcint output is
        # (Nj,Ni,3) in memory, so transpose the first two axes back to
        # (Ni,Nj,3) AO order. S is symmetric, so ∂S_ij/∂R_A = ∂S_ji/∂R_A
        # transposed, same trick `∇1e!` uses for its off-diagonal mirror.
        buf = zeros(Cdouble, 3*Ni*Nj)
        ∇overlap_ν!(buf, BS, i, j)
        out .= permutedims(reshape(buf, Nj, Ni, 3), (2,1,3))
    end
    return out
end

function ∇overlap(BS::BasisSet, A::Int, i::Int, j::Int)
    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])
    out = zeros(Ni, Nj, 3)
    return ∇overlap!(out, BS, A, i, j)
end

"""
    ∇overlap(BS::BasisSet, A) -> Array{Float64,3}
    ∇overlap(BS::BasisSet, A::Int, i::Int, j::Int) -> Array{Float64,3}

Gradient of the AO overlap matrix `S` w.r.t. atom `A`'s three Cartesian
coordinates, `∂S/∂R_A`, with `R_A` in bohr (see [Gradients](@ref) for
units).

# Methods

  - `∇overlap(BS, A)`: full, dense `nbas × nbas × 3` array.
  - `∇overlap(BS, A, i, j)`: just the `(Ni,Nj,3)` block for shells `i,j`
    of `BS` (shell indices, not AO indices).

For repeated calls, see `∇overlap!`, which writes into a preallocated
array instead of allocating.
"""
function ∇overlap(BS::BasisSet, A)
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇overlap!(out, BS, A)
end

∇overlap!(out, BS::BasisSet, A) = ∇1e!(∇overlap_μ!, out, BS, A)


###########################################################
###########################################################
#                        Kinetic                           #
###########################################################
###########################################################

"""
    ∇kinetic_μ!(out, BS::BasisSet{LCint}, i::Int, j::Int)

Same as [`∇overlap_μ!`](@ref) (same sign-flip reasoning, same lack of
bounds checking), for the kinetic energy operator instead of overlap.
"""
function ∇kinetic_μ!(out, BS::BasisSet{LCint}, i::Int, j::Int)
    cint1e_ipkin_sph!(out, [i, j], BS.lib)
    out .*= -1.0
    return out
end

"""
    ∇kinetic_ν!(out, BS::BasisSet{LCint}, i::Int, j::Int)

Same as [`∇kinetic_μ!`](@ref), differentiated with respect to shell `j`
(the "ν" AO) instead of `i`. Same lack of bounds checking applies.
"""
function ∇kinetic_ν!(out, BS::BasisSet{LCint}, i::Int, j::Int)
    cint1e_ipkin_sph!(out, [j, i], BS.lib)
    out .*= -1.0
    return out
end

"""
    ∇kinetic!(out, BS::BasisSet{LCint}, A::Int, i::Int, j::Int)
    ∇kinetic!(out, BS::BasisSet, A)

Mutating counterpart of [`∇kinetic`](@ref): writes into the caller-supplied
`out` instead of allocating. The shell-pair form is the primitive the
full-tensor form builds on -- see `∇overlap!` for the identical structure
and zero-block cases, `T` in place of `S`.

Only implemented for the `LCint` backend -- there is no `ACSint` fallback
for gradients.

# Methods

  - `∇kinetic!(out, BS, A, i, j)`: `out` must be `(Ni,Nj,3)`, the block
    for shells `i,j` of `BS` (shell indices, not AO indices).
  - `∇kinetic!(out, BS, A)`: `out` must be a dense `nbas × nbas × 3`
    array.
"""
function ∇kinetic!(out, BS::BasisSet{LCint}, A::Int, i::Int, j::Int)

    i_on_A, j_on_A = GaussianBasis.on_atom_flags(BS, A, i, j)

    if i_on_A == j_on_A
        out .= 0.0
        return out
    end

    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])

    if size(out) != (Ni, Nj, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Ni, $Nj, 3)"))
    end

    if i_on_A
        # Derivative lands on shell i (the one on atom A) directly -- raw
        # libcint output for a [i,j] call is already (Ni,Nj,3) in memory,
        # so write straight into `out`, no intermediate buffer needed.
        ∇kinetic_μ!(out, BS, i, j)
    else
        # Derivative lands on shell j (the one on atom A); compute via [j,i]
        # (j as the differentiated, first, shell) -- raw libcint output is
        # (Nj,Ni,3) in memory, so transpose the first two axes back to
        # (Ni,Nj,3) AO order. T is symmetric, so ∂T_ij/∂R_A = ∂T_ji/∂R_A
        # transposed, same trick `∇1e!` uses for its off-diagonal mirror.
        buf = zeros(Cdouble, 3*Ni*Nj)
        ∇kinetic_ν!(buf, BS, i, j)
        out .= permutedims(reshape(buf, Nj, Ni, 3), (2,1,3))
    end
    return out
end

function ∇kinetic(BS::BasisSet, A::Int, i::Int, j::Int)
    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])
    out = zeros(Ni, Nj, 3)
    return ∇kinetic!(out, BS, A, i, j)
end

"""
    ∇kinetic(BS::BasisSet, A) -> Array{Float64,3}
    ∇kinetic(BS::BasisSet, A::Int, i::Int, j::Int) -> Array{Float64,3}

Gradient of the AO kinetic energy matrix `T` w.r.t. atom `A`'s three
Cartesian coordinates, `∂T/∂R_A`, with `R_A` in bohr (see
[Gradients](@ref) for units).

# Methods

  - `∇kinetic(BS, A)`: full, dense `nbas × nbas × 3` array.
  - `∇kinetic(BS, A, i, j)`: just the `(Ni,Nj,3)` block for shells `i,j`
    of `BS` (shell indices, not AO indices).

For repeated calls, see `∇kinetic!`, which writes into a preallocated
array instead of allocating.
"""
function ∇kinetic(BS::BasisSet, A)
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇kinetic!(out, BS, A)
end

∇kinetic!(out, BS::BasisSet, A) = ∇1e!(∇kinetic_μ!, out, BS, A)


###########################################################
###########################################################
#                        Nuclear                           #
###########################################################
###########################################################

"""
    ∇nuclear_μ!(out, BS::BasisSet{LCint}, charge_atm, i::Int, j::Int)

Same as [`∇overlap_μ!`](@ref) (same sign-flip reasoning, same lack of
bounds checking), for a nuclear-attraction-type potential differentiated
with respect to shell `i` (the "μ" AO) instead -- except here the operator
itself is a potential that can move with a nucleus, not a fixed
multiplicative operator like `1` (overlap) or `-∇²/2` (kinetic).
`charge_atm` fixes WHICH potential: it's a libcint atom-description array
(same layout as `BS.lib.atm`, 6 `Cint` slots per atom) with a chosen subset
of nuclear charges zeroed out, so the raw libcint call sums `Zc/|r-Rc|`
only over the nuclei left un-zeroed. Like `∇overlap_μ!`/`∇kinetic_μ!`,
libcint's `cint1e_ipnuc_sph!` computes `∂/∂r` (w.r.t. the electron
coordinate), so this writes `-cint1e_ipnuc_sph!(...)`, i.e. `∂V^{charge}/∂R_i`
directly for the potential `charge_atm` encodes (see [Nuclear](@ref) for the
derivation).

> No bounds checking, no zero-block skipping, no output-size validation,
> and `charge_atm` isn't validated either -- same segfault/heap-corruption
> risks as [`∇overlap_μ!`](@ref) apply here too.
"""
function ∇nuclear_μ!(out, BS::BasisSet{LCint}, charge_atm, i::Int, j::Int)
    cint1e_ipnuc_sph!(out, Cint.([i-1, j-1]), charge_atm, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
    out .*= -1.0
    return out
end

"""
    ∇nuclear_ν!(out, BS::BasisSet{LCint}, charge_atm, i::Int, j::Int)

Same as [`∇nuclear_μ!`](@ref), differentiated with respect to shell `j`
(the "ν" AO) instead of `i`. Same lack of bounds checking applies.
"""
function ∇nuclear_ν!(out, BS::BasisSet{LCint}, charge_atm, i::Int, j::Int)
    cint1e_ipnuc_sph!(out, Cint.([j-1, i-1]), charge_atm, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
    out .*= -1.0
    return out
end

"""
    ∇nuclear!(out, BS::BasisSet{LCint}, A::Int, i::Int, j::Int)
    ∇nuclear!(out, BS::BasisSet, A)

Mutating counterpart of [`∇nuclear`](@ref): writes into the caller-supplied
`out` instead of allocating. The shell-pair form is the primitive the
full-tensor form builds on. Unlike `∇overlap!`/`∇kinetic!`, no case is ever
a free/skippable zero: `V_ij` sums the potential over every nucleus, so
even a shell pair with neither `i` nor `j` on atom `A` still has a
nonzero derivative through the `Z_A/|r-R_A|` operator term itself
moving. `K_A(i,j)`/`K_notA(i,j)` below are the bra-derivative
`cint1e_ipnuc_sph!` kernel at shell pair `(i,j)` using only atom `A`'s
charge / every charge except `A`'s:

  - `i,j` both on `A`:    `K_notA(i,j) + K_notA(j,i)ᵀ`
  - `i,j` both off `A`:    `-K_A(i,j) - K_A(j,i)ᵀ`
  - `i` on `A`, `j` off:    `K_notA(i,j) - K_A(j,i)ᵀ`
  - `i` off `A`, `j` on:     `-K_A(i,j) + K_notA(j,i)ᵀ`

(transpose over the two AO axes, per Cartesian direction). Prefer the
whole-array form when you need many shell pairs for the same atom -- this
one rebuilds the fudged nuclear-charge arrays on every call.

Only implemented for the `LCint` backend -- there is no `ACSint` fallback
for gradients.

# Methods

  - `∇nuclear!(out, BS, A, i, j)`: `out` must be `(Ni,Nj,3)`, the block
    for shells `i,j` of `BS` (shell indices, not AO indices).
  - `∇nuclear!(out, BS, A)`: `out` must be a dense `nbas × nbas × 3`
    array.
"""
function ∇nuclear!(out, BS::BasisSet{LCint}, A::Int, i::Int, j::Int; scratch=nothing)
    i_on_A, j_on_A = GaussianBasis.on_atom_flags(BS, A, i, j)
    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])

    if size(out) != (Ni, Nj, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Ni, $Nj, 3)"))
    end

    only_A = deepcopy(BS.lib.atm)
    no_A = deepcopy(BS.lib.atm)
    for k in eachindex(BS.atoms)
        if k == A
            no_A[1 + 6*(k-1)] = 0
        else
            only_A[1 + 6*(k-1)] = 0
        end
    end

    if i_on_A && j_on_A
        if scratch === nothing
            scratch = zeros(Ni, Nj, 3)
        end
        ∇nuclear_ν!(scratch, BS, no_A, i, j)
        out .+= permutedims(reshape(scratch, Nj, Ni, 3), (2,1,3))
        ∇nuclear_μ!(scratch, BS, no_A, i, j)
        out .+= reshape(scratch, Ni, Nj, 3)
        #out .= _nuc_pair_kernel(BS, no_A, i, j) .+ permutedims(_nuc_pair_kernel(BS, no_A, j, i), (2,1,3))
    elseif !i_on_A && !j_on_A
        out .= .-(_nuc_pair_kernel(BS, only_A, i, j) .+ permutedims(_nuc_pair_kernel(BS, only_A, j, i), (2,1,3)))
    elseif i_on_A && !j_on_A
        out .= _nuc_pair_kernel(BS, no_A, i, j) .- permutedims(_nuc_pair_kernel(BS, only_A, j, i), (2,1,3))
    else # !i_on_A && j_on_A
        out .= .-_nuc_pair_kernel(BS, only_A, i, j) .+ permutedims(_nuc_pair_kernel(BS, no_A, j, i), (2,1,3))
    end
    return out
end

function ∇nuclear(BS::BasisSet, A::Int, i::Int, j::Int)
    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])
    out = zeros(Ni, Nj, 3)
    return ∇nuclear!(out, BS, A, i, j)
end

# Bra-derivative cint1e_ipnuc_sph! kernel at shell pair (i,j), using a
# caller-supplied (possibly charge-fudged) atm array -- now a thin wrapper
# around ∇nuclear_μ!, which every call site here already matches (they get
# the "differentiated w.r.t. the FIRST shell argument" convention by
# swapping i,j themselves before calling). Kept only so ∇nuclear!'s branches
# below don't need touching yet; the goal is to inline ∇nuclear_μ!/
# ∇nuclear_ν! directly at each call site (mirroring ∇overlap!/∇kinetic!)
# and drop this wrapper entirely.
function _nuc_pair_kernel(BS::BasisSet{LCint}, charge_atm, i::Int, j::Int)
    Ni = num_basis(BS[i])
    Nj = num_basis(BS[j])
    out = zeros(Cdouble, Ni, Nj, 3)
    ∇nuclear_μ!(out, BS, charge_atm, i, j)
    return out
end

"""
    ∇nuclear(BS::BasisSet, A) -> Array{Float64,3}
    ∇nuclear(BS::BasisSet, A::Int, i::Int, j::Int) -> Array{Float64,3}

Gradient of the AO nuclear attraction matrix `V` w.r.t. atom `A`'s three
Cartesian coordinates, `∂V/∂R_A` (derivative of both the shell centers and
the potential itself, since moving atom `A` also moves its nuclear
charge), with `R_A` in bohr (see [Gradients](@ref) for units).

# Methods

  - `∇nuclear(BS, A)`: full, dense `nbas × nbas × 3` array.
  - `∇nuclear(BS, A, i, j)`: just the `(Ni,Nj,3)` block for shells `i,j`
    of `BS` (shell indices, not AO indices).

For repeated calls, see `∇nuclear!`, which writes into a preallocated
array instead of allocating.
"""
function ∇nuclear(BS::BasisSet, A)
    # Pre allocate output
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇nuclear!(out, BS, A)
end

function ∇nuclear!(out, BS::BasisSet, A)

    if size(out) != (BS.nbas, BS.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (nbas, nbas, 3)"))
    end

    atomA = BS.atoms[A]

    # Fudge lc_atoms
    only_A = deepcopy(BS.lib.atm)
    no_A = deepcopy(BS.lib.atm)
    for k = eachindex(BS.atoms)
        if k == A
            no_A[1 + 6*(k-1)] = 0
            continue
        end
        only_A[1 + 6*(k-1)] = 0
    end

    # Shell indexes for basis in the atom A (C notation: Starts from 0)
    Ashells = Int[]
    notAshells = Int[]
    for i in 1:BS.nshells
        b = BS.shells[i]
        if b.atom == atomA
            push!(Ashells, i)
        else
            push!(notAshells, i)
        end
    end

    Nvals = num_basis.(BS.shells)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    Nmax = maximum(Nvals)
    # i ∉ A & j ∉ A
    allocate(body) = body(zeros(Cdouble, 3*Nmax^2))
    workerpool(allocate, notAshells; chunksize = 1) do i, buf
        @inbounds begin
            Ni = Nvals[i]
            ioff = ao_offset[i]
            I = (ioff+1):(ioff+Ni)
            for j in notAshells
                Nj = Nvals[j]
                Nij = Ni*Nj
                joff = ao_offset[j]
                J = (joff+1):(joff+Nj)

                # + ⟨i'|Va|j⟩ + ⟨i|Va|j'⟩   (Note that Va is the potential of the nuclei A alone!!)
                cint1e_ipnuc_sph!(buf, Cint.([i-1,j-1]), only_A, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)

                # Get strides for each cartesian
                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    ∇k = reshape(buf[r], Int(Ni), Int(Nj))
                    out[I,J,k] .+= ∇k  # ⟨i'|Va|j ⟩
                end
            end
        end # inbounds
    end

    # i ∈ A & j ∈ A
    workerpool(allocate, Ashells; chunksize = 1) do i, buf
        @inbounds begin
            Ni = Nvals[i]
            ioff = ao_offset[i]
            I = (ioff+1):(ioff+Ni)
            for j in Ashells
                Nj = Nvals[j]
                Nij = Ni*Nj
                joff = ao_offset[j]
                J = (joff+1):(joff+Nj)

                # - ⟨i'|∑Vc|j⟩ - ⟨i|∑Vc|j'⟩ c != a
                cint1e_ipnuc_sph!(buf, Cint.([i-1,j-1]), no_A, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)

                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    ∇k = reshape(buf[r], Int(Ni), Int(Nj))
                    out[I,J,k] .-= ∇k  # ⟨i'|∑Vc|j ⟩ c != a
                end
            end
        end #inbounds
    end

    # i ∈ A & j ∉ A
    workerpool(allocate, Ashells; chunksize = 1) do i, buf
        @inbounds begin
            Ni = Nvals[i]
            ioff = ao_offset[i]
            I = (ioff+1):(ioff+Ni)
            for j in notAshells
                Nj = Nvals[j]
                Nij = Ni*Nj
                joff = ao_offset[j]
                J = (joff+1):(joff+Nj)

                # - ⟨i'|∑Vc|j⟩ + ⟨i|Va|j'⟩ c != a
                cint1e_ipnuc_sph!(buf, Cint.([i-1,j-1]), no_A, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    ∇k = buf[r]
                    out[I,J,k] .-= reshape(∇k, Int(Ni), Int(Nj))
                end

                cint1e_ipnuc_sph!(buf, Cint.([j-1,i-1]), only_A, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    ∇k = buf[r]
                    out[I,J,k] .+= transpose(reshape(∇k, Int(Nj), Int(Ni)))
                end
            end
        end #inbounds
    end

    # Add transpose values
    # This must be done outside the threaded loops
    # to avoid race conditions.
    for k in 1:3
        out[:,:,k] .+= out[:,:,k]'
    end

    return out
end


###########################################################
###########################################################
#                     General kernel                       #
###########################################################
###########################################################

# Shared whole-array kernel for overlap/kinetic, parametrized by `callback`
# (the bare, unchecked shell-differentiation primitive, e.g. `∇overlap_μ!`)
# instead of a string, mirroring how `get_1e_matrix!` in
# Integrals/OneElectron.jl takes a callback rather than branching on one.
# Calls the BARE primitive (`∇overlap_μ!`/`∇kinetic_μ!`), not the safe,
# bounds-checked `∇overlap!`/`∇kinetic!` shell-pair form -- going through
# the safe form would pay for a fresh `buf` allocation on every call, where
# this loop instead reuses one buffer per worker task across every shell
# pair it handles (translational invariance also lets it skip same-atom
# pairs and only visit Ashells×notAshells once, mirroring the rest via
# transpose). Safe here because `i`/`j` are always drawn from this loop's
# own bounded `Ashells`/`notAshells` and `buf` is sized from `Nmax` up
# front, not because the callback validates anything itself. Nuclear isn't
# wired through here at all: its three-loop structure amortizes the
# `deepcopy`-based fudged nuclear-charge arrays once across every shell
# pair, which this generic per-pair `callback` shape has no way to do.
function ∇1e(callback, BS::BasisSet, A)
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇1e!(callback, out, BS, A)
end

function ∇1e!(callback, out, BS::BasisSet, A)

    if size(out) != (BS.nbas, BS.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (nbas, nbas, 3)"))
    end

    atomA = BS.atoms[A]

    # Shell indexes for basis in the atom A
    Ashells = Int[]
    notAshells = Int[]
    for i in 1:BS.nshells
        b = BS.shells[i]
        if b.atom == atomA
            push!(Ashells, i)
        else
            push!(notAshells, i)
        end
    end

    Nvals = num_basis.(BS.shells)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    Nmax = maximum(Nvals)

    allocate(body) = body(zeros(Cdouble, 3*Nmax^2))
    workerpool(allocate, Ashells; chunksize = 1) do i, buf
        @inbounds begin
            Ni = Nvals[i]
            ioff = ao_offset[i]
            for j in notAshells
                Nj = Nvals[j]
                joff = ao_offset[j]
                Nij = Ni*Nj
                # Call the bare shell-differentiation primitive
                callback(buf, BS, i, j)
                I = (ioff+1):(ioff+Ni)
                J = (joff+1):(joff+Nj)

                # Get strides for each cartesian
                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    @views ∇k = buf[r]
                    out[I,J,k] .+= reshape(∇k, Ni, Nj)
                end

                # Copy over the transpose
                out[J,I,:] .+= permutedims(out[I,J,:], (2,1,3))
            end
        end #inbounds
    end
    return out
end
