function ∇ERI_2e4c(BS::BasisSet, iA)
    # Pre allocate output
    out = zeros(BS.nbas, BS.nbas, BS.nbas, BS.nbas, 3)
    return ∇ERI_2e4c!(out, BS, iA)
end

function ∇ERI_2e4c!(out, BS::BasisSet, iA)

    if size(out) != (BS.nbas, BS.nbas, BS.nbas, BS.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (N, N, N, N, 3)."))
    end

    A = BS.atoms[iA]

    # Shell indexes for basis in the atom A
    Ashells = Int[]
    notAshells = Int[]
    for i in 1:BS.nshells
        b = BS.basis[i]
        if b.atom == A
            push!(Ashells, i)
        else
            push!(notAshells, i)
        end
    end

    Nvals = num_basis.(BS.basis)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    Nmax = maximum(Nvals)
    #buf_arrays = [zeros(Cdouble, 3*Nmax^4) for _ = 1:Threads.nthreads()]


    # Find unique (i,j,k,l) combinations given permutational symmetry
    unique_idx = Tuple{Int, Int, Int, Int}[]
    for i = 1:BS.nshells
        for j = i:BS.nshells # i <= j
            for k = 1:BS.nshells
                for l = k:BS.nshells # k <= l
                    if index2(i-1,j-1) < index2(k-1,l-1)
                        continue
                    end
                    push!(unique_idx, (i,j,k,l))
                end
            end
        end
    end

    allocate(body) = body(zeros(Cdouble, 3*Nmax^4))
    workerpool(allocate, unique_idx; chunksize=10) do (i,j,k,l), buf
        x_in_A = map(in(Ashells), (i,j,k,l))

        # If no basis is centered on A, skip
        # If all basis are centered on A, skip
        if !any(x_in_A) || all(x_in_A)
            return
        end

        Ni = Nvals[i]
        Nj = Nvals[j]
        Nk = Nvals[k]
        Nl = Nvals[l]
        Nijkl = Ni*Nj*Nk*Nl

        ioff = ao_offset[i]
        joff = ao_offset[j]
        koff = ao_offset[k]
        loff = ao_offset[l]

        I = (ioff+1):(ioff+Ni)
        J = (joff+1):(joff+Nj)
        K = (koff+1):(koff+Nk)
        L = (loff+1):(loff+Nl)

        # [i'j|kl]
        if x_in_A[1]
            cint2e_ip1_sph!(buf, @SVector([i,j,k,l]), BS.lib)
            ∇q = reshape(view(buf, 1:3*Nijkl), Ni, Nj, Nk, Nl, 3)
            for q in 1:3
                @inbounds for (rl, nl) in enumerate(L), (rk, nk) in enumerate(K), (rj, nj) in enumerate(J), (ri, ni) in enumerate(I)
                    out[ni, nj, nk, nl, q] -= ∇q[ri, rj, rk, rl, q]
                end
            end
        end

        # [ij'|kl]
        if x_in_A[2]
            cint2e_ip1_sph!(buf, @SVector([j,i,k,l]), BS.lib)
            ∇q = reshape(view(buf, 1:3*Nijkl), Nj, Ni, Nk, Nl, 3)
            for q in 1:3
                @inbounds for (rl, nl) in enumerate(L), (rk, nk) in enumerate(K), (rj, nj) in enumerate(J), (ri, ni) in enumerate(I)
                    out[ni, nj, nk, nl, q] -= ∇q[rj, ri, rk, rl, q]
                end
            end
        end

        # [ij|k'l]
        if x_in_A[3]
            cint2e_ip1_sph!(buf, @SVector([k,l,i,j]), BS.lib)
            ∇q = reshape(view(buf, 1:3*Nijkl), Nk, Nl, Ni, Nj, 3)
            for q in 1:3
                @inbounds for (rl, nl) in enumerate(L), (rk, nk) in enumerate(K), (rj, nj) in enumerate(J), (ri, ni) in enumerate(I)
                    out[ni, nj, nk, nl, q] -= ∇q[rk, rl, ri, rj, q]
                end
            end
        end

        # [ij|kl']
        if x_in_A[4]
            cint2e_ip1_sph!(buf, @SVector([l,k,i,j]), BS.lib)
            ∇q = reshape(view(buf, 1:3*Nijkl), Nl, Nk, Ni, Nj, 3)
            for q in 1:3
                @inbounds for (rl, nl) in enumerate(L), (rk, nk) in enumerate(K), (rj, nj) in enumerate(J), (ri, ni) in enumerate(I)
                    out[ni, nj, nk, nl, q] -= ∇q[rl, rk, ri, rj, q]
                end
            end
        end

        for q in 1:3
            if i != j && k != l && index2(i,j) != index2(k,l)
                @inbounds for nl = L, nk = K, nj = J, ni = I
                    out[nj, ni, nk, nl, q] = out[ni, nj, nk, nl, q]
                    out[ni, nj, nl, nk, q] = out[ni, nj, nk, nl, q]
                    out[nj, ni, nl, nk, q] = out[ni, nj, nk, nl, q]
                    out[nk, nl, ni, nj, q] = out[ni, nj, nk, nl, q]
                    out[nl, nk, ni, nj, q] = out[ni, nj, nk, nl, q]
                    out[nk, nl, nj, ni, q] = out[ni, nj, nk, nl, q]
                    out[nl, nk, nj, ni, q] = out[ni, nj, nk, nl, q]
                end
            elseif k != l && index2(i,j) != index2(k,l)
                @inbounds for nl = L, nk = K, nj = J, ni = I
                    out[ni, nj, nl, nk, q] = out[ni, nj, nk, nl, q]
                    out[nk, nl, ni, nj, q] = out[ni, nj, nk, nl, q]
                    out[nl, nk, ni, nj, q] = out[ni, nj, nk, nl, q]
                end
            elseif i != j && index2(i,j) != index2(k,l)
                @inbounds for nl = L, nk = K, nj = J, ni = I
                    out[nj, ni, nk, nl, q] = out[ni, nj, nk, nl, q]
                    out[nk, nl, ni, nj, q] = out[ni, nj, nk, nl, q]
                    out[nk, nl, nj, ni, q] = out[ni, nj, nk, nl, q]
                end
            elseif i != j && k != l 
                @inbounds for nl = L, nk = K, nj = J, ni = I
                    out[nj, ni, nk, nl, q] = out[ni, nj, nk, nl, q]
                    out[ni, nj, nl, nk, q] = out[ni, nj, nk, nl, q]
                    out[nj, ni, nl, nk, q] = out[ni, nj, nk, nl, q]
                end
            elseif index2(i,j) != index2(k,l) 
                @inbounds for nl = L, nk = K, nj = J, ni = I
                    out[nk, nl, ni, nj, q] = out[ni, nj, nk, nl, q]
                end
            end
        end
    end

    return out
end

# --- Shell-quartet-level gradient ---
#
# GaussianBasis-idiomatic single-quartet primitive, mirroring the shell-pair
# primitives added for the 1-electron gradients (see OneElectronGrad.jl's
# header comment for the general rationale) -- motivated here by two things
# `∇ERI_2e4c!`'s own whole-array loop can't offer: (1) a building block for
# genuinely integral-direct gradient/CPHF code (compute one quartet,
# contract it immediately, discard -- unlike the plain energy ERI, which is
# reused across every SCF/CPHF iteration, a derivative quartet is only ever
# needed once to help form Jq/Kq or a Hessian block, so there's no caching
# benefit being given up by not materializing anything), and (2) avoiding
# GaussianBasis.Libcint's raw kernel calls at the call site the way Fermi.jl's
# `TwoElectronHess.jl` currently has to.
#
# Unlike `∇nuclear`, the bare Coulomb operator has no third "operator
# center" to differentiate -- so the free-zero case is exactly the same
# shape as overlap/kinetic's: a quartet's derivative is zero, no libcint
# call needed, whenever ALL FOUR shells share atom-membership status (all on
# atom `iA`, or all off it), by translational invariance. `∇ERI_2e4c!`'s
# own loop already exploits exactly this (`!any(x_in_A) || all(x_in_A)`,
# line 54 above).
#
# For a surviving quartet, this reuses `∇ERI_2e4c!`'s own per-shell
# derivative terms (`[i'j|kl]`, `[ij'|kl]`, `[ij|k'l]`, `[ij|kl']`, one
# `cint2e_ip1_sph!` call and index-permutation per shell that's actually on
# atom `iA`) directly, for an ARBITRARY shell ordering -- unlike the
# whole-array loop, this function does not need `i≤j`, `k≤l`, or any
# canonical-quartet reordering first: that reordering in `∇ERI_2e4c!` is
# purely a performance optimization (compute an 8-fold-symmetric block once,
# propagate it to every symmetric position instead of recomputing), not a
# correctness requirement -- the underlying `[i'j|kl]+[ij'|kl]+[ij|k'l]+
# [ij|kl']` formula is already well-defined for any `(i,j,k,l)` on its own.

"""
    ∇ERI_2e4c(BS::BasisSet, iA::Int, i::Int, j::Int, k::Int, l::Int)
    ∇ERI_2e4c(BS::BasisSet, on_A::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)

Two forms. The `iA::Int` form is the convenient, standalone-safe one:
computes `on_A` (which of shells `i,j,k,l` sit on atom `iA`, via `===` --
see `on_atom_flags`) and returns the free zero (no libcint call) when
`i,j,k,l` are ALL on atom `iA` or ALL off it. The `on_A::NTuple{4,Bool}`
form is the unchecked core: it does whatever `on_A` says unconditionally,
no membership computation and no zero-skip, for callers (like Fermi.jl's
gradient loop) that have already screened at a higher level and precomputed
`on_A` once outside a hot loop -- calling this form on an all-on/all-off
`on_A` does NOT raise an error, it just wastefully computes an answer of
exactly zero the long way, so getting that screening right is entirely the
caller's responsibility for this form.
"""
function ∇ERI_2e4c(BS::BasisSet, iA::Int, i::Int, j::Int, k::Int, l::Int)
    on_A = on_atom_flags(BS, iA, i, j, k, l)
    Ni, Nj, Nk, Nl = num_basis(BS.basis[i]), num_basis(BS.basis[j]), num_basis(BS.basis[k]), num_basis(BS.basis[l])
    out = zeros(Ni, Nj, Nk, Nl, 3)
    (!any(on_A) || all(on_A)) && return out
    return ∇ERI_2e4c!(out, BS, on_A, i, j, k, l)
end

function ∇ERI_2e4c!(out, BS::BasisSet, iA::Int, i::Int, j::Int, k::Int, l::Int)
    on_A = on_atom_flags(BS, iA, i, j, k, l)
    if !any(on_A) || all(on_A)
        out .= 0.0
        return out
    end
    return ∇ERI_2e4c!(out, BS, on_A, i, j, k, l)
end

function ∇ERI_2e4c(BS::BasisSet, on_A::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)
    Ni, Nj, Nk, Nl = num_basis(BS.basis[i]), num_basis(BS.basis[j]), num_basis(BS.basis[k]), num_basis(BS.basis[l])
    out = zeros(Ni, Nj, Nk, Nl, 3)
    return ∇ERI_2e4c!(out, BS, on_A, i, j, k, l)
end

function ∇ERI_2e4c!(out, BS::BasisSet, on_A::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)
    Ni, Nj, Nk, Nl = num_basis(BS.basis[i]), num_basis(BS.basis[j]), num_basis(BS.basis[k]), num_basis(BS.basis[l])
    Nijkl = Ni*Nj*Nk*Nl
    buf = zeros(Cdouble, 3*Nijkl)
    tmp = zeros(Cdouble, 3*Nijkl)
    shls = zeros(Cint, 4)
    return ∇ERI_2e4c!(out, BS, on_A, i, j, k, l, buf, tmp, shls)
end

"""
    ∇ERI_2e4c!(out, BS::BasisSet, on_A::NTuple{4,Bool}, i, j, k, l, buf, tmp, shls)

Scratch-buffer-accepting core: identical math to the 7-argument form above,
but takes caller-owned `buf`/`tmp` (each sized `>= 3*Nmax^4`, `Nmax` = the
largest `num_basis` over any shell the caller will ever pass) and `shls`
(sized 4) instead of allocating them fresh every call. Exists because this
function sits in the innermost loop of Fermi.jl's integral-direct gradient
(called once per (atom, canonical-quartet) visit, millions of times for a
real molecule) -- profiling that loop found the allocating form costing
~1.3 KB/call (a fresh `buf`, plus one allocating `permutedims` per non-first
branch), which adds up to several GB of GC churn over a full gradient. This
form is zero-allocation: `permutedims!` (in-place, same semantics as
`permutedims`, just no fresh output array) replaces `permutedims`, and `buf`/
`shls` are reused in place instead of rebuilt per call. Safe for concurrent
use as long as each caller (e.g. each worker task) has its own `buf`/`tmp`/
`shls` -- these are mutated in place and not thread-safe to share.
"""
function ∇ERI_2e4c!(out, BS::BasisSet, on_A::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int,
                     buf::Vector{Cdouble}, tmp::Vector{Cdouble}, shls::Vector{Cint})

    Ni, Nj, Nk, Nl = num_basis(BS.basis[i]), num_basis(BS.basis[j]), num_basis(BS.basis[k]), num_basis(BS.basis[l])

    if size(out) != (Ni, Nj, Nk, Nl, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Ni, $Nj, $Nk, $Nl, 3)."))
    end

    out .= 0.0

    Nijkl = Ni*Nj*Nk*Nl
    lib = BS.lib
    bufv = view(buf, 1:3*Nijkl)
    tmpv = view(tmp, 1:3*Nijkl)

    # [i'j|kl]
    if on_A[1]
        shls[1] = i-1; shls[2] = j-1; shls[3] = k-1; shls[4] = l-1
        cint2e_ip1_sph!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
        out .-= reshape(bufv, Ni, Nj, Nk, Nl, 3)
    end

    # [ij'|kl]
    if on_A[2]
        shls[1] = j-1; shls[2] = i-1; shls[3] = k-1; shls[4] = l-1
        cint2e_ip1_sph!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
        ∇q = reshape(bufv, Nj, Ni, Nk, Nl, 3)
        dest = reshape(tmpv, Ni, Nj, Nk, Nl, 3)
        permutedims!(dest, ∇q, (2,1,3,4,5))
        out .-= dest
    end

    # [ij|k'l]
    if on_A[3]
        shls[1] = k-1; shls[2] = l-1; shls[3] = i-1; shls[4] = j-1
        cint2e_ip1_sph!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
        ∇q = reshape(bufv, Nk, Nl, Ni, Nj, 3)
        dest = reshape(tmpv, Ni, Nj, Nk, Nl, 3)
        permutedims!(dest, ∇q, (3,4,1,2,5))
        out .-= dest
    end

    # [ij|kl']
    if on_A[4]
        shls[1] = l-1; shls[2] = k-1; shls[3] = i-1; shls[4] = j-1
        cint2e_ip1_sph!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
        ∇q = reshape(bufv, Nl, Nk, Ni, Nj, 3)
        dest = reshape(tmpv, Ni, Nj, Nk, Nl, 3)
        permutedims!(dest, ∇q, (3,4,2,1,5))
        out .-= dest
    end

    return out
end

"""
    schwarz_bounds(BS::BasisSet)

Per-shell-pair Cauchy-Schwarz screening bound `σ_ij := sqrt(max|(ij|ij)|)`,
computed from the ordinary (undifferentiated) integrals -- an O(nshells^2)
pass over diagonal shell quartets, independent of any nuclear derivative.
Used by both `sparseERI_2e4c` and `∇sparseERI_2e4c` to skip shell quartets
whose bound falls below a cutoff. `∇sparseERI_2e4c` is typically called once
per atom (a full gradient loops over every atom, `eri_grad_JK` does the same
inside CPHF); since this bound doesn't depend on which atom is being
differentiated, callers making several `∇sparseERI_2e4c` calls should
compute it once via this function and pass it through via the `ij_vals`/
`σvals` kwargs, instead of paying this cost again for every atom.
"""
function schwarz_bounds(BS::BasisSet)
    Nvals = num_basis.(BS.basis)
    Nmax = maximum(Nvals)
    num_ij = Int((BS.nshells^2 - BS.nshells)/2) + BS.nshells
    ij_vals = Array{NTuple{2,Int32}}(undef, num_ij)
    σvals = zeros(Cdouble, num_ij)
    tmp = zeros(Cdouble, Nmax^4)
    for i = 1:BS.nshells
        for j = i:BS.nshells
            idx = index2(i-1,j-1) + 1
            ij_vals[idx] = (i,j)
            ERI_2e4c!(tmp, BS, i, i, j, j)
            σvals[idx] = √maximum(tmp)
        end
    end
    return ij_vals, σvals
end

"""
    ∇sparseERI_2e4c(BS::BasisSet, iA, cutoff=1e-12; ij_vals=nothing, σvals=nothing)

Derivative (w.r.t. atom `iA`'s three Cartesian directions) of the unique
(permutation-compressed) two-electron four-center integrals, screened the
same way `sparseERI_2e4c` screens the energy integrals: a shell quartet is
skipped up front whenever `σ_ij*σ_kl <= cutoff` (see `schwarz_bounds`). This
is a valid proxy for the derivative too -- differentiating a Gaussian-product
ERI w.r.t. a nuclear center only introduces a bounded polynomial prefactor
via the chain rule, the exponential shell-pair falloff with distance that
makes the bound work for the energy integral is unchanged for its
derivative. Quartets entirely on atom `iA` or entirely off it are also
skipped (exactly zero by translational invariance, not a numerical cutoff).

`ij_vals`/`σvals` default to a fresh `schwarz_bounds(BS)` call if not
supplied -- pass a precomputed pair in when calling this repeatedly across
atoms (see `schwarz_bounds`'s docstring) to avoid recomputing it every time.

Results are accumulated into a growable buffer (`sizehint!`ed from a cheap
upper-bound estimate on the number of surviving elements, no extra integral
evaluations needed for the estimate) rather than a fixed array pre-sized at
the full theoretical unique-element count -- peak memory scales with the
number of surviving elements instead of nbas^4/8. Unlike `sparseERI_2e4c`
(called once per SCF run, amortizing its own screening/threading cost over
every Fock build that follows), this is typically called once per atom per
gradient/Hessian evaluation -- there's much less work to amortize a
thread-pool's synchronization overhead over, and measurement backed up the
theory: at default (single-)thread count, threading this made it ~2x
*slower* than a plain loop for a small molecule, so this stays serial.
"""
function ∇sparseERI_2e4c(BS::BasisSet, iA, cutoff = 1e-12; ij_vals = nothing, σvals = nothing)

    A = BS.atoms[iA]

    # Shell indexes for basis in the atom A
    in_A = falses(BS.nshells)
    for i in 1:BS.nshells
        BS.basis[i].atom == A && (in_A[i] = true)
    end

    # Pre compute a list of number of basis for each shell (2l +1)
    Nvals = num_basis.(BS.basis)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    Nmax = maximum(Nvals)

    # Unique shell pairs with i ≤ j
    num_ij = Int((BS.nshells^2 - BS.nshells)/2) + BS.nshells

    if ij_vals === nothing || σvals === nothing
        ij_vals, σvals = schwarz_bounds(BS)
    end

    # Candidate shell quartets (ij,kl), ij ≤ kl: Schwarz-screened AND
    # touching atom A in some but not all of the four slots. A plain nested
    # loop rather than a filtered/flattened generator -- eltype() on the
    # latter gives up and infers Any (see sparseERI_2e4c's comment on the
    # same issue), which would box every (i,j,k,l) tuple in the hot loop
    # below; measured as the dominant cost here, more than any of the actual
    # per-quartet integral work.
    #
    # Upper bound on the number of surviving AO elements (Nvals products
    # only, no ERI calls), used to sizehint! the buffers below.
    size_ub = 0
    @inbounds for ij in 1:num_ij
        i, j = ij_vals[ij]
        for kl in ij:num_ij
            σvals[ij]*σvals[kl] <= cutoff && continue
            k, l = ij_vals[kl]
            nA = in_A[i] + in_A[j] + in_A[k] + in_A[l]
            (nA == 0 || nA == 4) && continue
            size_ub += Nvals[i]*Nvals[j]*Nvals[k]*Nvals[l]
        end
    end

    indexes = sizehint!(Vector{NTuple{4,Int16}}(), size_ub)
    ∇x = sizehint!(Cdouble[], size_ub)
    ∇y = sizehint!(Cdouble[], size_ub)
    ∇z = sizehint!(Cdouble[], size_ub)

    buf = zeros(Cdouble, 3*Nmax^4)

    # i,j,k,l => Shell indexes starting at one
    # I, J, K, L => AO indexes starting at one
    @inbounds for ij in 1:num_ij
        i, j = ij_vals[ij]
        for kl in ij:num_ij
            σvals[ij]*σvals[kl] <= cutoff && continue
            k, l = ij_vals[kl]
            nA = in_A[i] + in_A[j] + in_A[k] + in_A[l]
            (nA == 0 || nA == 4) && continue
        begin
            Ni, Nj, Nk, Nl = Nvals[i], Nvals[j], Nvals[k], Nvals[l]
            Nij = Ni*Nj
            Nijk = Nij*Nk
            Nijkl = Nijk*Nl
            ioff, joff, koff, loff = ao_offset[i], ao_offset[j], ao_offset[k], ao_offset[l]

            # NOTE: Using loops instead of array operations could make this more efficient
            # Compute ERI
            bufx = zeros(Cdouble, Ni, Nj, Nk, Nl)
            bufy = zeros(Cdouble, Ni, Nj, Nk, Nl)
            bufz = zeros(Cdouble, Ni, Nj, Nk, Nl)
            if in_A[i]
                cint2e_ip1_sph!(buf, [i,j,k,l], BS.lib)
                bufx += reshape(buf[1:Nijkl], Ni, Nj, Nk, Nl)
                bufy += reshape(buf[Nijkl+1:2*Nijkl], Ni, Nj, Nk, Nl)
                bufz += reshape(buf[2*Nijkl+1:3*Nijkl], Ni, Nj, Nk, Nl)
            end

            if in_A[j]
                cint2e_ip1_sph!(buf, [j,i,k,l], BS.lib)
                bufx += permutedims(reshape(buf[1:Nijkl],           Nj, Ni, Nk, Nl), (2,1,3,4))
                bufy += permutedims(reshape(buf[Nijkl+1:2*Nijkl],   Nj, Ni, Nk, Nl), (2,1,3,4))
                bufz += permutedims(reshape(buf[2*Nijkl+1:3*Nijkl], Nj, Ni, Nk, Nl), (2,1,3,4))
            end

            if in_A[k]
                cint2e_ip1_sph!(buf, [k,l,i,j], BS.lib)
                bufx += permutedims(reshape(buf[1:Nijkl],           Nk, Nl, Ni, Nj), (3,4,1,2))
                bufy += permutedims(reshape(buf[Nijkl+1:2*Nijkl],   Nk, Nl, Ni, Nj), (3,4,1,2))
                bufz += permutedims(reshape(buf[2*Nijkl+1:3*Nijkl], Nk, Nl, Ni, Nj), (3,4,1,2))
            end

            if in_A[l]
                cint2e_ip1_sph!(buf, [l,k,i,j], BS.lib)
                bufx += permutedims(reshape(buf[1:Nijkl],           Nl, Nk, Ni, Nj), (3,4,2,1))
                bufy += permutedims(reshape(buf[Nijkl+1:2*Nijkl],   Nl, Nk, Ni, Nj), (3,4,2,1))
                bufz += permutedims(reshape(buf[2*Nijkl+1:3*Nijkl], Nl, Nk, Ni, Nj), (3,4,2,1))
            end

            ### This block aims to retrieve unique elements within buf and map them to AO indexes
            # is, js, ks, ls are indexes within the shell e.g. for a p shell is = (1, 2, 3)
            # bl, bkl, bjkl are used to map the (i,j,k,l) index into a one-dimensional index for buf
            # That is, get the correct integrals for the AO quartet.
            #
            # Only when the same shell-pair is reused for bra and ket (i==k && j==l)
            # does this block become self-symmetric (is,js and ks,ls range over the
            # exact same configuration space), so every (I,J,K,L) generated below has
            # a mirror (K,L,I,J) also generated in this same shell quartet -- guarded
            # by IJ>KL&&continue to push each AO quartet once, not twice.
            self_paired = i == k && j == l
            for ls = 1:Nl
                L = loff + ls
                bl = Nijk*(ls-1)
                for ks = 1:Nk
                    K = koff + ks
                    L < K && break

                    # L ≥ K
                    # index2 for K,L
                    KL = ((L * (L - 1)) ÷ 2) + K - 1

                    bkl = Nij*(ks-1) + bl
                    for js = 1:Nj
                        J = joff + js
                        bjkl = Ni*(js-1) + bkl
                        for is = 1:Ni
                            I = ioff + is
                            J < I && break

                            # index2 for I,J
                            IJ = ((J * (J - 1)) ÷ 2) + I - 1

                            self_paired && IJ > KL && continue

                            push!(∇x, -bufx[is + bjkl])
                            push!(∇y, -bufy[is + bjkl])
                            push!(∇z, -bufz[is + bjkl])
                            push!(indexes, KL ≥ IJ ? (I, J, K, L) : (K, L, I, J))
                        end
                    end
                end
            end
        end
        end
    end

    return indexes, ∇x, ∇y, ∇z
end

function ∇ERI_2e3c(BS1::BasisSet, BS2::BasisSet, iA)
    # Pre allocate output
    out = zeros(BS1.nbas, BS1.nbas, BS2.nbas, 3)
    return ∇ERI_2e3c!(out, BS1, BS2, iA)
end

function ∇ERI_2e3c!(out, BS1::BasisSet, BS2::BasisSet, iA)

    atoms = unique(vcat(BS1.atoms, BS2.atoms))
    basis = vcat(BS1.basis, BS2.basis)

    Bmerged = BasisSet("$(BS1.name*BS2.name)", atoms, basis)

    if size(out) != (BS1.nbas, BS1.nbas, BS2.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (N1, N1, N2, 3)."))
    end

    A = BS1.atoms[iA]

    # Shell indexes for basis 1 in the atom A
    Ashells1 = Int[]
    notAshells1 = Int[]
    for i in 1:BS1.nshells
        b = BS1.basis[i]
        if b.atom == A
            push!(Ashells1, i)
        else
            push!(notAshells1, i)
        end
    end

    # Shell indexes for basis 2 in the atom A
    Ashells2 = Int[]
    notAshells2 = Int[]
    for i in 1:BS2.nshells
        b = BS2.basis[i]
        if b.atom == A
            push!(Ashells2, i)
        else
            push!(notAshells2, i)
        end
    end

    Nvals1 = num_basis.(BS1.basis)
    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    Nmax1 = maximum(Nvals1)

    Nvals2 = num_basis.(BS2.basis)
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]
    Nmax2 = maximum(Nvals2)

    buf = zeros(Cdouble, 3*Nmax1^2*Nmax2)

    for i = 1:BS1.nshells
        for j = i:BS1.nshells # i <= j
            for k = 1:BS2.nshells

                x_in_A = [i in Ashells1, j in Ashells1, k in Ashells2]

                # If no basis is centered on A, skip
                # If all basis are centered on A, skip
                if !any(x_in_A) || all(x_in_A)
                    continue
                end

                Ni = Nvals1[i]
                Nj = Nvals1[j]
                Nk = Nvals2[k]
                Nijk = Ni*Nj*Nk

                ioff = ao_offset1[i]
                joff = ao_offset1[j]
                koff = ao_offset2[k]

                I = (ioff+1):(ioff+Ni)
                J = (joff+1):(joff+Nj)
                K = (koff+1):(koff+Nk)

	        # [i'j|k]
                if x_in_A[1]
                    cint3c2e_ip1_sph!(buf, [i,j,k+BS1.nshells], Bmerged.lib)
                    for q in 1:3
                        r = (1+Nijk*(q-1)):(q*Nijk)
                        ∇q = reshape(buf[r], Int(Ni), Int(Nj), Int(Nk))
                        out[I,J,K,q] += -∇q
                    end
                end

                # [ij'|k]
                if x_in_A[2]
                    cint3c2e_ip1_sph!(buf, [j,i,k+BS1.nshells], Bmerged.lib)
                    for q in 1:3
                        r = (1+Nijk*(q-1)):(q*Nijk)
                        ∇q = reshape(buf[r], Int(Nj), Int(Ni), Int(Nk))
                        out[I,J,K,q] += -permutedims(∇q, (2,1,3))
                    end
                end

                # [ij|k']
                if x_in_A[3]
                    cint3c2e_ip2_sph!(buf, [i,j,k+BS1.nshells], Bmerged.lib)
                    for q in 1:3
                        r = (1+Nijk*(q-1)):(q*Nijk)
                        ∇q = reshape(buf[r], Int(Ni), Int(Nj), Int(Nk))
                        out[I,J,K,q] += -∇q
                    end
                end

                if i != j
                    for q in 1:3
                        # i,j permutation
                        out[J, I, K, q] .= permutedims(out[I, J, K, q], (2,1,3))
                    end
                end
            end
        end
    end

    return out
end

function ∇ERI_2e2c(BS::BasisSet, iA)
    # Pre allocate output
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇ERI_2e2c!(out, BS, iA)
end

function ∇ERI_2e2c!(out, BS::BasisSet, iA)

    if size(out) != (BS.nbas, BS.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (N, N, 3)."))
    end

    A = BS.atoms[iA]

    # Shell indexes for basis in the atom A
    Ashells = Int[]
    notAshells = Int[]
    for i in 1:BS.nshells
        b = BS.basis[i]
        if b.atom == A
            push!(Ashells, i)
        else
            push!(notAshells, i)
        end
    end

    Nvals = num_basis.(BS.basis)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    Nmax = maximum(Nvals)

    buf = zeros(Cdouble, 3*Nmax^2)

    for i = 1:BS.nshells
        for j = i:BS.nshells # i <= j

            x_in_A = [x in Ashells for x = (i,j)]

            # If no basis is centered on A, skip
            # If all basis are centered on A, skip
            if !any(x_in_A) || all(x_in_A)
                continue
            end

            Ni = Nvals[i]
            Nj = Nvals[j]
            Nij = Ni*Nj

            ioff = ao_offset[i]
            joff = ao_offset[j]

            I = (ioff+1):(ioff+Ni)
            J = (joff+1):(joff+Nj)

            # [i'|j]
            if x_in_A[1]
                cint2c2e_ip1_sph!(buf, [i,j], BS.lib)
                for q in 1:3
                    r = (1+Nij*(q-1)):(q*Nij)
                    ∇q = reshape(buf[r], Int(Ni), Int(Nj))
                    out[I,J,q] += -∇q
                end
            end

            # [i|j']
            if x_in_A[2]
                cint2c2e_ip1_sph!(buf, [j,i], BS.lib)
                for q in 1:3
                    r = (1+Nij*(q-1)):(q*Nij)
                    ∇q = reshape(buf[r], Int(Nj), Int(Ni))
                    out[I,J,q] += -permutedims(∇q, (2,1))
                end
            end

            if i != j
                for q in 1:3
                    # i,j permutation
                    out[J, I, q] .= permutedims(out[I, J, q], (2,1))
                end
            end
        end
    end

    return out
end
