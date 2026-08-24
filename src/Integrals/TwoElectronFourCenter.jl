"""
    sparseERI_2e4c(BS::BasisSet, cutoff=1e-12) -> (indexes, values)

Compute the unique, permutationally non-redundant two-electron four-center
integrals `(ij|kl)` for `BS` (chemist's notation), applying Cauchy-Schwarz
shell-pair screening and discarding any integral with `abs(value) <= cutoff`.

Returns a pair `(indexes, values)`: `indexes` is a `Vector{NTuple{4,Int16}}`
of `(I,J,K,L)` AO indices (1-based) and `values` the corresponding
`Vector{Float64}` of integral values, `indexes[n]` paired with `values[n]`.
Only one representative of each permutationally-equivalent AO quartet is
returned. Element order is not sorted or otherwise guaranteed (it depends on
how the underlying work is split across threads) -- callers that need a
canonical unique integral once (e.g. Fock builds) don't care about order,
but shouldn't rely on any particular one either.
"""
function sparseERI_2e4c(BS::BasisSet, cutoff = 1e-12)
    # Pre compute a list of angular momentum numbers (l) for each shell
    Nvals = num_basis.(BS.shells)
    Nmax = maximum(Nvals)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset = cumsum(Nvals) .- Nvals

    # Unique shell pairs (i<=j), ordered so that ij_vals[n] is the pair with
    # index2(i-1,j-1) == n-1 -- σvals below is indexed by that same composite
    # index, so σvals[n] is the screening factor for ij_vals[n].
    ij_vals = unique_ij(BS.nshells)
    num_ij = length(ij_vals)

    # Cauchy-Schwarz screening factors σ_ij = sqrt(max |(ij|ij)|) over the
    # shell-pair block, giving the bound |(ij|kl)| <= σ_ij * σ_kl, so a whole
    # shell quartet can be skipped when σ_ij*σ_kl <= cutoff.
    #
    # (ij|ij) -- shells (i,j,i,j) -- is the integral the bound is built from.
    # Note the block maximum is legitimate here and equals the maximum over the
    # block's diagonal elements (mu nu|mu nu): the (ij|ij) block is a Gram
    # matrix in the Coulomb metric, so by Cauchy-Schwarz no off-diagonal
    # element can exceed the largest diagonal one.
    # Threaded: these are O(nshells^2) independent shell-quartet evaluations
    # writing to disjoint σvals entries. Left serial this was ~11% of the whole
    # routine's runtime at 24 threads, capping its speedup by Amdahl.
    σvals = zeros(Cdouble, num_ij)
    allocate_σ(body) = body(zeros(Cdouble, Nmax^4))
    workerpool(allocate_σ, 1:num_ij; chunksize=10) do idx, tmp
        @inbounds begin
            i, j = ij_vals[idx]
            nblk = (Nvals[i]*Nvals[j])^2
            ERI_2e4c!(tmp, BS, i, j, i, j)
            # abs, and only over the nblk entries this call actually wrote --
            # `tmp` is Nmax^4 long and keeps stale data from earlier iterations
            # beyond that point.
            σvals[idx] = √maximum(abs, view(tmp, 1:nblk))
        end
    end

    # Surviving shell quartets, together with an upper bound on the number of
    # AO elements they can contribute (from block sizes alone, no ERI calls).
    # Built in a single pass: this used to be a lazy flatten-of-filtered-
    # generators that was walked twice -- once to accumulate size_ub, once to
    # fill the channel -- re-running the screening test both times, and whose
    # eltype() inferred as Any. A concrete Vector fixes both.
    #
    # size_ub is used only to sizehint! the per-task buffers below: geometric
    # Vector growth is amortized O(1), but still copies ~2x the final size in
    # total over the doublings, so hinting close to the true size avoids that
    # overshoot (measured ~35% less peak memory on a dense system with the hint).
    ijkl_vals = NTuple{4,Int16}[]
    size_ub = 0
    @inbounds for kl in 1:num_ij
        k, l = ij_vals[kl]
        σkl = σvals[kl]
        for ij in 1:kl
            σvals[ij] * σkl > cutoff || continue
            i, j = ij_vals[ij]
            push!(ijkl_vals, (i, j, k, l))
            size_ub += Nvals[i]*Nvals[j]*Nvals[k]*Nvals[l]
        end
    end

    # Results are not written into a shared nbas-sized dense buffer (whose size
    # would depend only on nbas, not on how many quartets survive screening).
    # Each task instead accumulates into its own private growable buffer, sized
    # upfront via the estimate above, and buffers are concatenated into
    # right-sized output arrays once every task is done. Peak memory then scales
    # with the number of surviving elements instead of nbas^4.
    ntasks = Threads.nthreads()
    chunksize = 10

    # Partitioning a Vector yields views, so the channel is typed from the
    # partition iterator itself rather than hardcoding a chunk type.
    chunks = Iterators.partition(ijkl_vals, chunksize)
    requests = Channel{eltype(chunks)}(Inf)
    for chunk in chunks
        put!(requests, chunk)
    end
    close(requests)

    per_task_hint = cld(size_ub, ntasks)
    task_vals = [sizehint!(Cdouble[], per_task_hint) for _ = 1:ntasks]
    task_idxs = [sizehint!(Vector{NTuple{4,Int16}}(), per_task_hint) for _ = 1:ntasks]

    @sync for t = 1:ntasks
        Threads.@spawn begin
            buf = zeros(Cdouble, Nmax^4)
            vals = task_vals[t]
            idxs = task_idxs[t]
            for chunk in requests
                for (i,j,k,l) in chunk
                    @inbounds begin
                        Li, Lj, Lk, Ll = map(n->Nvals[n], (i,j,k,l))
                        Lij = Li*Lj
                        Lijk = Lij*Lk

                        ioff, joff, koff, loff = map(n->ao_offset[n], (i,j,k,l))

                        # Only when the same shell-pair is reused for bra and ket does this
                        # block become self-symmetric (is,js and ks,ls range over the exact
                        # same configuration space), guaranteeing every (I,J,K,L) has a mirror
                        # (K,L,I,J) also enumerated below. Elsewhere IJ vs KL ordering carries
                        # no such redundancy and both must be kept.
                        self_paired = i == k && j == l

                        # Compute ERI
                        ERI_2e4c!(buf, BS, i, j, k ,l)

                        # is, js, ks, ls are indexes within the shell e.g. for a p shell is = (1, 2, 3)
                        # bl, bkl, bjkl are used to map the (i,j,k,l) index into a one-dimensional index for buf
                        for ls = 1:Ll
                            L = loff + ls
                            bl = Lijk*(ls-1)
                            for ks = 1:Lk
                                K = koff + ks
                                L < K && break

                                bkl = Lij*(ks-1) + bl
                                for js = 1:Lj
                                    J = joff + js
                                    bjkl = Li*(js-1) + bkl
                                    for is = 1:Li
                                        I = ioff + is
                                        J < I && break

                                        if self_paired
                                            IJ = (J * (J + 1)) >> 1 + I
                                            KL = (L * (L + 1)) >> 1 + K
                                            IJ > KL && continue
                                        end

                                        v = buf[is + bjkl]
                                        if abs(v) > cutoff
                                            push!(vals, v)
                                            push!(idxs, (I, J, K, L))
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    total = sum(length, task_vals)
    out = Vector{Cdouble}(undef, total)
    indexes = Vector{NTuple{4,Int16}}(undef, total)

    offset = 0
    for t = 1:ntasks
        n = length(task_vals[t])
        n == 0 && continue
        copyto!(out, offset+1, task_vals[t], 1, n)
        copyto!(indexes, offset+1, task_idxs[t], 1, n)
        offset += n
    end

    return indexes, out
end

function ERI_2e4c(BS::BasisSet, i, j, k, l)
    out = zeros(eltype(BS.atoms[1].xyz), num_basis(BS.shells[i]), num_basis(BS.shells[j]),
                num_basis(BS.shells[k]), num_basis(BS.shells[l]))
    ERI_2e4c!(out, BS, i, j, k, l)
    return out
end

# Mutating, shell-quartet-level form of ERI_2e4c(BS, i, j, k, l): writes into
# a caller-supplied `out` (sized (Ni,Nj,Nk,Nl)) instead of allocating.
# Dispatches on the integral backend (LCint vs. the ACSint fallback below).
"""
    ERI_2e4c!(out, BS::BasisSet, i, j, k, l)
    ERI_2e4c!(out, BS::BasisSet)

Mutating counterpart of [`ERI_2e4c`](@ref): writes into the caller-supplied
`out` instead of allocating. This shell-quartet form is the primitive the
full-tensor form builds on.

# Methods

  - `ERI_2e4c!(out, BS, i, j, k, l)`: `out` must be `(Ni,Nj,Nk,Nl)`, the
    `(ij|kl)` block (chemist's notation) for shells `i,j,k,l` of `BS` (shell
    indices, not AO indices).
  - `ERI_2e4c!(out, BS)`: `out` must be a dense
    `nbas × nbas × nbas × nbas` array.
"""
function ERI_2e4c!(out, BS::BasisSet{LCint}, i, j, k, l)
    cint2e_sph!(out, @SVector([i,j,k,l]), BS.lib)
end

function ERI_2e4c!(out, BS::BasisSet, i, j, k, l)
    generate_ERI_quartet!(out, BS, i, j, k, l)
end

"""
    ERI_2e4c(BS::BasisSet) -> Array{Float64,4}
    ERI_2e4c(BS::BasisSet, i, j, k, l) -> Array{Float64,4}

Compute the two-electron four-center integral tensor `(ij|kl)` (chemist's
notation), respecting the standard 8-fold permutational symmetry
(`(ij|kl)=(ji|kl)=(ij|lk)=(kl|ij)=...`).

# Methods

  - `ERI_2e4c(BS)`: full, dense `nbas × nbas × nbas × nbas` tensor for `BS`.
    For large basis sets prefer `sparseERI_2e4c`, which screens and stores
    only the unique elements.
  - `ERI_2e4c(BS, i, j, k, l)`: just the `(Ni,Nj,Nk,Nl)` block for shells
    `i,j,k,l` of `BS` (shell indices, not AO indices).

For repeated calls (e.g. in a hot loop), see `ERI_2e4c!`, which writes into
a preallocated array instead of allocating.
"""
function ERI_2e4c(BS::BasisSet)
    N = BS.nbas
    out = zeros(N, N, N, N)
    ERI_2e4c!(out, BS)
end

function ERI_2e4c!(out, BS::BasisSet)
    # NOTE: `out` is deliberately not zeroed. Every unique quartet below is
    # computed and scattered to all of its symmetry images, so every element
    # of `out` is written exactly once and any prior contents are fully
    # overwritten. Anything that makes the quartet loop skip work -- notably
    # Cauchy-Schwarz screening, which `sparseERI_2e4c` does but this dense
    # build intentionally does not -- breaks that invariant and MUST add a
    # `fill!(out, 0.0)` here, or screened blocks will silently retain the
    # caller's stale data.

    # Save a list containing the number of basis for each shell
    Nvals = num_basis.(BS.shells)
    Nmax = maximum(Nvals)

    # Get slice corresponding to the address in S where the compute chunk goes
    ao_offset = cumsum(Nvals) .- Nvals
    ranges = [(ao_offset[i]+1):(ao_offset[i]+Nvals[i]) for i = 1:BS.nshells]

    # Find unique (i,j,k,l) combinations given permutational symmetry
    unique_idx = unique_ijkl(BS.nshells)

    # Initialize array for results
    allocate(body) = body(zeros(Cdouble, Nmax^4))
    workerpool(allocate, unique_idx; chunksize=10) do (id,jd,kd,ld), buf
        # unique_ijkl yields 1-based shell indexes; the symmetry bookkeeping
        # below is written against libcint's 0-based convention.
        i, j, k, l = id-1, jd-1, kd-1, ld-1
        Ni, Nj, Nk, Nl = Nvals[id], Nvals[jd], Nvals[kd], Nvals[ld]

        # Compute ERI
        ERI_2e4c!(buf, BS, id, jd, kd, ld)

        # Move results to output array
        ri, rj, rk, rl = ranges[id], ranges[jd], ranges[kd], ranges[ld]
        out[ri, rj, rk, rl] .= reshape(@view(buf[1:Ni*Nj*Nk*Nl]), (Ni, Nj, Nk, Nl))

        if i != j && k != l && index2(i,j) != index2(k,l)
            @inbounds for ni = ri, nj = rj, nk = rk, nl = rl
                out[nj, ni, nk, nl] = out[ni, nj, nk, nl]
                out[ni, nj, nl, nk] = out[ni, nj, nk, nl]
                out[nj, ni, nl, nk] = out[ni, nj, nk, nl]
                out[nk, nl, ni, nj] = out[ni, nj, nk, nl]
                out[nl, nk, ni, nj] = out[ni, nj, nk, nl]
                out[nk, nl, nj, ni] = out[ni, nj, nk, nl]
                out[nl, nk, nj, ni] = out[ni, nj, nk, nl]
            end
        elseif k != l && index2(i,j) != index2(k,l)
            @inbounds for ni = ri, nj = rj, nk = rk, nl = rl
                out[ni, nj, nl, nk] = out[ni, nj, nk, nl]
                out[nk, nl, ni, nj] = out[ni, nj, nk, nl]
                out[nl, nk, ni, nj] = out[ni, nj, nk, nl]
            end
        elseif i != j && index2(i,j) != index2(k,l)
            @inbounds for ni = ri, nj = rj, nk = rk, nl = rl
                out[nj, ni, nk, nl] = out[ni, nj, nk, nl]
                out[nk, nl, ni, nj] = out[ni, nj, nk, nl]
                out[nk, nl, nj, ni] = out[ni, nj, nk, nl]
            end
        elseif i != j && k != l
            @inbounds for ni = ri, nj = rj, nk = rk, nl = rl
                out[nj, ni, nk, nl] = out[ni, nj, nk, nl]
                out[ni, nj, nl, nk] = out[ni, nj, nk, nl]
                out[nj, ni, nl, nk] = out[ni, nj, nk, nl]
            end
        elseif index2(i,j) != index2(k,l)
            @inbounds for ni = ri, nj = rj, nk = rk, nl = rl
                out[nk, nl, ni, nj] = out[ni, nj, nk, nl]
            end
        end
    end #sync

    return out
end
