function ∇1e(BS::BasisSet, compute::String, iA)
    # Pre allocate output
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇1e!(out, BS, compute, iA)
end

function ∇1e!(out, BS::BasisSet, compute::String, iA)

    if size(out) != (BS.nbas, BS.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (nbas, nbas, 3)"))
    end

    if compute == "overlap"
        libcint_1e! =  cint1e_ipovlp_sph!
    elseif compute == "kinetic"
        libcint_1e! =  cint1e_ipkin_sph!
    elseif compute == "nuclear"
        return ∇nuclear(BS, iA)
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

    allocate(body) = body(zeros(Cdouble, 3*Nmax^2))
    workerpool(allocate, Ashells; chunksize = 1) do i, buf
        @inbounds begin
            Ni = Nvals[i]
            ioff = ao_offset[i]
            for j in notAshells
                Nj = Nvals[j]
                joff = ao_offset[j]
                Nij = Ni*Nj
                # Call libcint
                libcint_1e!(buf, [i,j], BS.lib)
                I = (ioff+1):(ioff+Ni)
                J = (joff+1):(joff+Nj)

                # Get strides for each cartesian
                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    @views ∇k = buf[r]
                    #∇k = buf[r]
                    out[I,J,k] .-= reshape(∇k, Ni, Nj)
                end

                # Copy over the transpose
                out[J,I,:] .+= permutedims(out[I,J,:], (2,1,3))
            end
        end #inbounds
    end
    return out
end

∇overlap(BS::BasisSet, iA) = ∇1e(BS, "overlap", iA)
∇overlap!(out, BS::BasisSet, iA) = ∇1e!(out, BS, "overlap", iA)
∇kinetic(BS::BasisSet, iA) = ∇1e(BS, "kinetic", iA)
∇kinetic!(out, BS::BasisSet, iA) = ∇1e!(out, BS, "kinetic", iA)

# --- Shell-pair-level gradients ---
#
# GaussianBasis-idiomatic (hides libcint's shell indexing/buffer-layout
# plumbing the same way `overlap!(out, BS, i, j)` already does), but at the
# finer, single-shell-pair granularity of the plain integrals rather than
# `∇overlap(BS,iA)`'s whole-array granularity. NOT wired into `∇1e!`/
# `∇nuclear!` above (both keep their own translational-invariance-based
# Ashells/notAshells loop untouched) -- `∇nuclear!`'s three-loop structure in
# particular amortizes the `deepcopy`-based fudged nuclear-charge arrays
# once across every shell pair, and a naive per-pair call to the function
# below would pay that cost again on every single call, which the array
# assembler has no reason to do when it already has the cheaper path.
#
# Derived by symbolically tracing the existing, already-validated `∇1e!`/
# `∇nuclear!` loops (matching the transpose-mirroring/final-symmetrization
# they do) for an arbitrary shell pair (P,Q) in either order, rather than
# re-deriving the underlying integral calculus from scratch -- lower risk of
# introducing a NEW error, though the nuclear case (four branches, derived
# by hand from a fairly intricate three-loop-plus-final-symmetrize source)
# is still checked against `∇nuclear(BS,iA)` in the test suite before being
# trusted.

"""
    ∇overlap(BS::BasisSet, iA::Int, i::Int, j::Int)

Shell-pair-level overlap gradient: `∂S_ij/∂R_iA` for shells `i,j` w.r.t.
atom `iA`'s three Cartesian directions, as an `(Ni,Nj,3)` block -- the same
values `∇overlap(BS,iA)` would place at that shell pair's AO block. Exactly
zero, returned without any libcint call, whenever `i,j` are BOTH on atom
`iA` or BOTH off it: `S_ij` doesn't depend on `R_iA` at all if neither
shell sits there, and translating `i,j` together with atom `iA` (both
belong to it) leaves their relative separation -- and so `S_ij` -- unchanged.
"""
∇overlap(BS::BasisSet, iA::Int, i::Int, j::Int) = ∇1e_pair(BS, "overlap", iA, i, j)
∇overlap!(out, BS::BasisSet, iA::Int, i::Int, j::Int) = ∇1e_pair!(out, BS, "overlap", iA, i, j)

"""
    ∇kinetic(BS::BasisSet, iA::Int, i::Int, j::Int)

Shell-pair-level kinetic energy gradient -- see `∇overlap(BS,iA,i,j)`,
identical structure and zero-block cases, `T` in place of `S`.
"""
∇kinetic(BS::BasisSet, iA::Int, i::Int, j::Int) = ∇1e_pair(BS, "kinetic", iA, i, j)
∇kinetic!(out, BS::BasisSet, iA::Int, i::Int, j::Int) = ∇1e_pair!(out, BS, "kinetic", iA, i, j)

function ∇1e_pair(BS::BasisSet, compute::String, iA::Int, i::Int, j::Int)
    Ni = num_basis(BS.basis[i])
    Nj = num_basis(BS.basis[j])
    out = zeros(Ni, Nj, 3)
    return ∇1e_pair!(out, BS, compute, iA, i, j)
end

function ∇1e_pair!(out, BS::BasisSet, compute::String, iA::Int, P::Int, Q::Int)

    if compute == "overlap"
        libcint_1e! = cint1e_ipovlp_sph!
    elseif compute == "kinetic"
        libcint_1e! = cint1e_ipkin_sph!
    else
        throw(ArgumentError("compute must be \"overlap\" or \"kinetic\", got \"$compute\""))
    end

    A = BS.atoms[iA]
    P_on_A = BS.basis[P].atom == A
    Q_on_A = BS.basis[Q].atom == A
    Np = num_basis(BS.basis[P])
    Nq = num_basis(BS.basis[Q])

    if size(out) != (Np, Nq, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Np, $Nq, 3)"))
    end

    if P_on_A == Q_on_A
        out .= 0.0
        return out
    end

    if P_on_A
        # Derivative lands on shell P (the one on atom A) directly.
        buf = zeros(Cdouble, 3*Np*Nq)
        libcint_1e!(buf, [P, Q], BS.lib)
        for k in 1:3
            r = (1+Np*Nq*(k-1)):(k*Np*Nq)
            out[:,:,k] .= .-reshape(buf[r], Np, Nq)
        end
    else
        # Derivative lands on shell Q (the one on atom A); compute via [Q,P]
        # (Q as the differentiated, first, shell) and transpose back to
        # (P,Q) AO order -- S is symmetric, so ∂S_PQ/∂R_A = ∂S_QP/∂R_A
        # transposed, same trick `∇1e!` uses for its off-diagonal mirror.
        buf = zeros(Cdouble, 3*Nq*Np)
        libcint_1e!(buf, [Q, P], BS.lib)
        for k in 1:3
            r = (1+Nq*Np*(k-1)):(k*Nq*Np)
            out[:,:,k] .= permutedims(.-reshape(buf[r], Nq, Np), (2,1))
        end
    end
    return out
end

function ∇nuclear(BS::BasisSet, iA)
    # Pre allocate output
    out = zeros(BS.nbas, BS.nbas, 3)
    return ∇nuclear!(out, BS, iA)
end

function ∇nuclear!(out, BS::BasisSet, iA)

    if size(out) != (BS.nbas, BS.nbas, 3)
        throw(DimensionMismatch("Size of the output array needs to be (nbas, nbas, 3)"))
    end

    A = BS.atoms[iA]

    # Fudge lc_atoms
    lc_atoms_A = deepcopy(BS.lib.atm)
    lc_atoms_woA = deepcopy(BS.lib.atm)
    for i = eachindex(BS.atoms)
        if i == iA
            lc_atoms_woA[1 + 6*(i-1)] = 0
            continue 
        end
        lc_atoms_A[1 + 6*(i-1)] = 0
    end

    # Shell indexes for basis in the atom A (C notation: Starts from 0)
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
                cint1e_ipnuc_sph!(buf, Cint.([i-1,j-1]), lc_atoms_A, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)

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
                cint1e_ipnuc_sph!(buf, Cint.([i-1,j-1]), lc_atoms_woA, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)

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
                cint1e_ipnuc_sph!(buf, Cint.([i-1,j-1]), lc_atoms_woA, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
                for k in 1:3
                    r = (1+Nij*(k-1)):(k*Nij)
                    ∇k = buf[r]
                    out[I,J,k] .-= reshape(∇k, Int(Ni), Int(Nj))
                end

                cint1e_ipnuc_sph!(buf, Cint.([j-1,i-1]), lc_atoms_A, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
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

"""
    ∇nuclear(BS::BasisSet, iA::Int, P::Int, Q::Int)

Shell-pair-level nuclear attraction gradient: `∂V_PQ/∂R_iA` for shells
`P,Q` w.r.t. atom `iA`'s three Cartesian directions, as an `(Np,Nq,3)`
block -- the same values `∇nuclear(BS,iA)` would place at that shell pair's
AO block.

Unlike `∇overlap`/`∇kinetic`, no case is ever a free/skippable zero: `V_PQ`
sums the potential over every nucleus, so even a shell pair with neither
`P` nor `Q` on atom `iA` still has a nonzero derivative through the
`Z_iA/|r-R_iA|` operator term itself moving. Four cases instead, mirroring
`∇nuclear!`'s three-loop-plus-final-symmetrize structure exactly (derived
by tracing that already-validated code for an arbitrary pair, see this
file's header comment) -- `K_A(p,q)`/`K_notA(p,q)` below are the bra-derivative
`cint1e_ipnuc_sph!` kernel at shell pair `(p,q)` using only atom `iA`'s
charge / every charge except `iA`'s, respectively:

  - P,Q both on `iA`:    `-K_notA(P,Q) - K_notA(Q,P)ᵀ`
  - P,Q both off `iA`:    `K_A(P,Q) + K_A(Q,P)ᵀ`
  - P on `iA`, Q off:    `-K_notA(P,Q) + K_A(Q,P)ᵀ`
  - P off `iA`, Q on:     `K_A(P,Q) - K_notA(Q,P)ᵀ`

(transpose over the two AO axes, per `k`). Builds its own fudged
nuclear-charge arrays per call -- fine for a one-off shell pair, but
`∇nuclear!`'s whole-array loop builds them once and reuses them across
every pair instead of calling this function `nshells^2` times.
"""
function ∇nuclear(BS::BasisSet, iA::Int, P::Int, Q::Int)
    Np = num_basis(BS.basis[P])
    Nq = num_basis(BS.basis[Q])
    out = zeros(Np, Nq, 3)
    return ∇nuclear!(out, BS, iA, P, Q)
end

function ∇nuclear!(out, BS::BasisSet, iA::Int, P::Int, Q::Int)
    A = BS.atoms[iA]
    P_on_A = BS.basis[P].atom == A
    Q_on_A = BS.basis[Q].atom == A
    Np = num_basis(BS.basis[P])
    Nq = num_basis(BS.basis[Q])

    if size(out) != (Np, Nq, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Np, $Nq, 3)"))
    end

    lc_atoms_A = deepcopy(BS.lib.atm)
    lc_atoms_woA = deepcopy(BS.lib.atm)
    for i in eachindex(BS.atoms)
        if i == iA
            lc_atoms_woA[1 + 6*(i-1)] = 0
        else
            lc_atoms_A[1 + 6*(i-1)] = 0
        end
    end

    if P_on_A && Q_on_A
        out .= .-(_nuc_pair_kernel(BS, lc_atoms_woA, P, Q) .+ permutedims(_nuc_pair_kernel(BS, lc_atoms_woA, Q, P), (2,1,3)))
    elseif !P_on_A && !Q_on_A
        out .= _nuc_pair_kernel(BS, lc_atoms_A, P, Q) .+ permutedims(_nuc_pair_kernel(BS, lc_atoms_A, Q, P), (2,1,3))
    elseif P_on_A && !Q_on_A
        out .= .-_nuc_pair_kernel(BS, lc_atoms_woA, P, Q) .+ permutedims(_nuc_pair_kernel(BS, lc_atoms_A, Q, P), (2,1,3))
    else # !P_on_A && Q_on_A
        out .= _nuc_pair_kernel(BS, lc_atoms_A, P, Q) .- permutedims(_nuc_pair_kernel(BS, lc_atoms_woA, Q, P), (2,1,3))
    end
    return out
end

# Bra-derivative cint1e_ipnuc_sph! kernel at shell pair (p,q), using a
# caller-supplied (possibly charge-fudged) atm array, reshaped from libcint's
# flat 3-chunk buffer layout into an (Np,Nq,3) block.
function _nuc_pair_kernel(BS::BasisSet, charge_atm, p::Int, q::Int)
    Np = num_basis(BS.basis[p])
    Nq = num_basis(BS.basis[q])
    Npq = Np*Nq
    buf = zeros(Cdouble, 3*Npq)
    cint1e_ipnuc_sph!(buf, Cint.([p-1,q-1]), charge_atm, BS.lib.natm, BS.lib.bas, BS.lib.nbas, BS.lib.env)
    out = zeros(Np, Nq, 3)
    for k in 1:3
        r = (1+Npq*(k-1)):(k*Npq)
        out[:,:,k] .= reshape(buf[r], Np, Nq)
    end
    return out
end
