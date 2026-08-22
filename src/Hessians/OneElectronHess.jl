function ∇21e(BS::BasisSet, compute::String, iA, iB)
    out = zeros(BS.nbas, BS.nbas, 3, 3)
    return ∇21e!(out, BS, compute, iA, iB)
end

# Second derivative of overlap/kinetic w.r.t. two nuclear positions (atoms iA,
# iB). Unlike nuclear attraction, these integrals only depend on two
# positions total (the two shell centers), so -- for a given shell pair (i,j)
# with atom(i)=Ai, atom(j)=Aj -- the only combinations of (dR_iA, dR_iB) that
# can be nonzero are built from whichever of shell i's/shell j's centers
# happen to coincide with atom iA and/or atom iB:
#   - both derivatives land on shell i (Ai==iA and Ai==iB, i.e. iA==iB==Ai):
#     same-center kernel (ipip) on shell i, i.e. cint1e_ipip*_sph!([i,j],...)
#   - one derivative on shell i, one on shell j (cross terms): cross-center
#     kernel (ipXip) on [i,j]
#   - both derivatives land on shell j: same-center kernel on shell j,
#     obtained by calling it with shells reversed ([j,i]) and transposing
#     back into (i,j) AO ordering
# When iA==iB and both shells sit on that atom, all four combinations apply
# simultaneously and add together (chain rule for a doubled variable), which
# fixes the size of the diagonal Hessian block.
function ∇21e!(out, BS::BasisSet, compute::String, iA, iB)

    if size(out) != (BS.nbas, BS.nbas, 3, 3)
        throw(DimensionMismatch("Size of the output array needs to be (nbas, nbas, 3, 3)"))
    end
    out .= 0.0

    if compute == "overlap"
        ipip! = cint1e_ipipovlp_sph!
        ipXip! = cint1e_ipovlpip_sph!
    elseif compute == "kinetic"
        ipip! = cint1e_ipipkin_sph!
        ipXip! = cint1e_ipkinip_sph!
    elseif compute == "nuclear_shellonly"
        # Whole-molecule (all nuclei summed, unweighted by *which* atom pair we
        # target) piece of the nuclear-attraction Hessian where both
        # derivatives land on shell centers. This is only PART of
        # ∇2nuclear! -- the remaining piece, where at least one derivative
        # lands on a nuclear *charge* position, is added separately in
        # NuclearHess.jl via isolated per-nucleus rinv terms (to avoid
        # double-counting this shell-only contribution).
        ipip! = cint1e_ipipnuc_sph!
        ipXip! = cint1e_ipnucip_sph!
    elseif compute == "metric"
        # 2-center two-electron auxiliary metric (P|Q), density fitting's
        # J_PQ. Depends on exactly two positions (the two shell centers,
        # same as overlap/kinetic, no third "operator" position like nuclear
        # attraction), so it reuses this exact same-shell/cross-shell
        # structure -- the only difference is which basis set `BS` is (the
        # auxiliary one) and which kernel family. `cint2c2e_*` kernels take
        # an extra trailing `opt` argument (2-electron kernels always do,
        # unlike 1-electron ones) but that's handled inside the Libcint.jl
        # binding itself, invisible here.
        ipip! = cint2c2e_ipip1_sph!
        ipXip! = cint2c2e_ip1ip2_sph!
    else
        throw(ArgumentError("compute must be \"overlap\", \"kinetic\", \"nuclear_shellonly\", or \"metric\" (use ∇2nuclear! for the full nuclear attraction Hessian)"))
    end

    Aat = BS.atoms[iA]
    Bat = BS.atoms[iB]

    Nvals = num_basis.(BS.basis)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    Nmax = maximum(Nvals)
    buf = zeros(Cdouble, 9*Nmax^2)
    # Reused across every shell pair below (each of size <= Nmax^2*9) instead
    # of allocating `block`/a permuted temporary fresh per pair -- same
    # rationale as the 2-electron scratch-buffer fixes in TwoElectronGrad.jl/
    # TwoElectronFourCenterHess.jl, just at O(nshells^2) instead of
    # O(nshells^4) calls (smaller per-call footprint, but a real allocation
    # source over a full Hessian's worth of atom pairs nonetheless).
    block = zeros(Cdouble, Nmax, Nmax, 3, 3)
    permbuf = zeros(Cdouble, Nmax, Nmax, 3, 3)
    shls = zeros(Cint, 2)

    @inbounds for i in 1:BS.nshells
        atom_i = BS.basis[i].atom
        X_i = (atom_i == Aat)
        Y_i = (atom_i == Bat)
        # No early-exit on X_i||Y_i alone here: shell i not touching A or B is
        # still relevant if shell j does (e.g. i on some other atom, j on A --
        # contributes the single-shell "d2_jj" term below).

        Ni = Nvals[i]
        ioff = ao_offset[i]
        I = (ioff+1):(ioff+Ni)

        for j in 1:BS.nshells
            atom_j = BS.basis[j].atom
            X_j = (atom_j == Aat)
            Y_j = (atom_j == Bat)

            (X_i || X_j) || continue
            (Y_i || Y_j) || continue

            Nj = Nvals[j]
            joff = ao_offset[j]
            J = (joff+1):(joff+Nj)
            Nij = Ni*Nj
            lib = BS.lib

            blockv = view(block, 1:Ni, 1:Nj, :, :)
            blockv .= 0.0

            if X_i && Y_i
                shls[1] = i-1; shls[2] = j-1
                bufv = view(buf, 1:9*Nij)
                ipip!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
                blockv .+= reshape(bufv, Ni, Nj, 3, 3)
            end

            if (X_i && Y_j) || (X_j && Y_i)
                # cint1e_ip*ip_sph!'s two derivative-component axes come out
                # in (ket,bra) order, not (bra,ket): raw[:,:,q,p] is
                # d2X/d(shell i)_p d(shell j)_q. For overlap/kinetic this is
                # invisible (those integrals depend only on the shell-shell
                # separation, so their cross-Hessian is provably symmetric
                # under the p<->q swap, making raw and its transpose
                # numerically identical) -- but not in general (e.g. nuclear
                # attraction's shell-only piece, which also has a fixed
                # background charge distribution breaking that symmetry), so
                # the two placements need genuinely different orientations:
                # X_i&&Y_j wants d2/d(shell i=A)_m d(shell j=B)_k = raw[:,:,k,m]
                # X_j&&Y_i wants d2/d(shell j=A)_m d(shell i=B)_k = raw[:,:,m,k]
                shls[1] = i-1; shls[2] = j-1
                bufv = view(buf, 1:9*Nij)
                ipXip!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
                raw = reshape(bufv, Ni, Nj, 3, 3)
                if X_i && Y_j
                    permbufv = reshape(view(permbuf, 1:9*Nij), Ni, Nj, 3, 3)
                    permutedims!(permbufv, raw, (1,2,4,3))
                    blockv .+= permbufv
                end
                X_j && Y_i && (blockv .+= raw)
            end

            if X_j && Y_j
                shls[1] = j-1; shls[2] = i-1
                bufv = view(buf, 1:9*Nij)
                ipip!(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
                blockT = reshape(bufv, Nj, Ni, 3, 3)
                permbufv = reshape(view(permbuf, 1:9*Nij), Ni, Nj, 3, 3)
                permutedims!(permbufv, blockT, (2,1,3,4))
                blockv .+= permbufv
            end

            out[I,J,:,:] .+= blockv
        end
    end

    return out
end

"""
    ∇2overlap(BS::BasisSet, iA, iB) -> Array{Float64,4}

Second derivative (Hessian) of the AO overlap matrix `S` w.r.t. atoms
`iA`,`iB`'s three Cartesian coordinates each, `∂²S/∂R_iA∂R_iB`, with
`R_iA`/`R_iB` in bohr (see [Hessians](@ref) for units). Returns a dense
`nbas × nbas × 3 × 3` array. For repeated calls, see `∇2overlap!`.
"""
∇2overlap(BS::BasisSet, iA, iB) = ∇21e(BS, "overlap", iA, iB)
∇2overlap!(out, BS::BasisSet, iA, iB) = ∇21e!(out, BS, "overlap", iA, iB)

"""
    ∇2kinetic(BS::BasisSet, iA, iB) -> Array{Float64,4}

Second derivative (Hessian) of the AO kinetic energy matrix `T` w.r.t.
atoms `iA`,`iB`'s three Cartesian coordinates each, `∂²T/∂R_iA∂R_iB`, with
`R_iA`/`R_iB` in bohr (see [Hessians](@ref) for units). Returns a dense
`nbas × nbas × 3 × 3` array. For repeated calls, see `∇2kinetic!`.
"""
∇2kinetic(BS::BasisSet, iA, iB) = ∇21e(BS, "kinetic", iA, iB)
∇2kinetic!(out, BS::BasisSet, iA, iB) = ∇21e!(out, BS, "kinetic", iA, iB)

"""
    ∇2ERI_2e2c(auxbset::BasisSet, iA, iB)

Second derivative (Hessian, atoms `iA`,`iB`, with `R_iA`/`R_iB` in bohr --
see [Hessians](@ref) for units) of the 2-center two-electron auxiliary
metric `(P|Q)` (density fitting's `J_PQ`). Output `(naux,naux,3,3)`.
Reuses `∇21e!`'s same-shell/cross-shell structure via the `"metric"`
compute-case -- see this file's header comment.
"""
∇2ERI_2e2c(BS::BasisSet, iA, iB) = ∇21e(BS, "metric", iA, iB)
∇2ERI_2e2c!(out, BS::BasisSet, iA, iB) = ∇21e!(out, BS, "metric", iA, iB)
