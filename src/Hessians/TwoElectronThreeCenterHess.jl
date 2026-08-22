# Second derivative of the 3-center two-electron integral (μν|P) w.r.t. two
# nuclear positions (atoms iA, iB). Unlike overlap/kinetic (2 shell
# positions) or the 4-center ERI (4 shell positions but bra/ket-paired),
# this integral has 3 shell positions with no pairing symmetry between them:
# μ, ν (both "regular" basis shells, symmetric under μ<->ν swap since
# (μν|P)=(νμ|P)) and P (the auxiliary/fitting shell). That gives 3
# same-shell placements (μμ, νν, PP) and 3 distinct cross placements (μν,
# μP, νP) -- 6 total instead of the 2-shell case's 2 (same, cross).
#
# Kernel-to-placement mapping (confirmed against libcint's own cint_funcs.h
# doc comments, mirroring exactly how Fermi.jl's 4-center ERI Hessian
# (TwoElectronHess.jl) already worked this out for cint2e_ipvip1/ip1ip2 --
# same naming logic, one derivative order down in shell count):
#
#   cint3c2e_ipip1  both derivatives on shell-argument-position-1 (μ, by
#                   default -- shell-argument order can be permuted to put
#                   ν there instead, same trick as the first-derivative
#                   ∇ERI_2e3c! already uses)
#   cint3c2e_ipip2  both derivatives on the auxiliary shell (libcint's own
#                   "position 2" for 3c2e, matching how the existing
#                   first-derivative cint3c2e_ip1/ip2 already split
#                   "regular" vs "auxiliary" rather than literal shell-arg
#                   position 1 vs 2)
#   cint3c2e_ipvip1 one derivative each on μ and ν -- the "same group"
#                   cross term (both regular shells, no aux involved)
#   cint3c2e_ip1ip2 one derivative each on a regular shell and the
#                   auxiliary shell -- the "opposite group" cross term
#
# Same as the 4-center case's ipvip1/ip1ip2, the two derivative-component
# axes of a CROSS kernel's raw output come out reversed from the naive
# (position-1-component, position-2-component) expectation -- ipip-type
# (same-shell) kernels need no such correction, since mixed partials of the
# same point commute. eri3c_hess_kernel applies that correction
# unconditionally (harmless no-op for the symmetric same-shell case,
# mirroring Fermi.jl's eri_hess_kernel exactly) so every other function here
# can assume normalized (position-1-component, position-2-component) axis
# order without re-deriving the correction per call site.
#
# Validated: ∇2FD_ERI_2e3c (central difference of the already-trusted
# ∇ERI_2e3c), translational invariance (sum over all iB of the Hessian
# block, for fixed iA, is zero -- checked to machine precision), and an
# independent cross-check of the μν cross term against the value implied by
# translational invariance applied to the already-validated μμ/νν/μP/νP
# terms (d2f/dμdν = -d2f/dμ² - d2f/dμdP, from differentiating the
# first-derivative translational-invariance identity dμ+dν+dP=0 once more
# w.r.t. μ) -- an independent check on cint3c2e_ipvip1 specifically that
# doesn't depend on finite difference at all.

function eri3c_hess_kernel(kern, merged::BasisSet, shells)
    Na, Nb, Nc = num_basis.(merged.shells[collect(shells)])
    buf = zeros(Cdouble, 9 * Na * Nb * Nc)
    kern(buf, shells, merged.lib)
    return permutedims(reshape(buf, Na, Nb, Nc, 3, 3), (1, 2, 3, 5, 4))
end

# Same-shell second derivative: both derivatives land on the shell at
# position posK (1=μ, 2=ν, 3=aux). Returned in (μ,ν,aux) AO-axis order.
function eri3c_hess_same(merged::BasisSet, i, j, kshell, posK::Int)
    if posK == 1
        eri3c_hess_kernel(cint3c2e_ipip1_sph!, merged, [i, j, kshell])
    elseif posK == 2
        raw = eri3c_hess_kernel(cint3c2e_ipip1_sph!, merged, [j, i, kshell])
        permutedims(raw, (2, 1, 3, 4, 5))
    elseif posK == 3
        eri3c_hess_kernel(cint3c2e_ipip2_sph!, merged, [i, j, kshell])
    else
        throw(ArgumentError("posK must be 1, 2, or 3"))
    end
end

# Cross second derivative: one derivative on the shell at position posA, one
# on the shell at position posB (posA != posB). Returned in (μ,ν,aux)
# AO-axis order, derivative-component axes (4,5) in (posA-component,
# posB-component) order.
function eri3c_hess_cross(merged::BasisSet, i, j, kshell, posA::Int, posB::Int)
    swp = posA > posB
    a, b = swp ? (posB, posA) : (posA, posB)

    d = if (a, b) == (1, 2)
        eri3c_hess_kernel(cint3c2e_ipvip1_sph!, merged, [i, j, kshell])
    elseif (a, b) == (1, 3)
        eri3c_hess_kernel(cint3c2e_ip1ip2_sph!, merged, [i, j, kshell])
    elseif (a, b) == (2, 3)
        raw = eri3c_hess_kernel(cint3c2e_ip1ip2_sph!, merged, [j, i, kshell])
        permutedims(raw, (2, 1, 3, 4, 5))
    else
        throw(ArgumentError("unexpected position pair ($a,$b)"))
    end

    # d(posB,posA)[...,x,y] = d(posA,posB)[...,y,x] (mixed partials commute)
    return swp ? permutedims(d, (1, 2, 3, 5, 4)) : d
end

"""
    ∇2ERI_2e3c(BS1::BasisSet, BS2::BasisSet, iA, iB)

Second derivative (Hessian, atoms `iA`,`iB`, with `R_iA`/`R_iB` in bohr --
see [Hessians](@ref) for units) of the 3-center two-electron integral
`(μν|P)` (`BS1`=regular basis, `BS2`=auxiliary/fitting basis -- density
fitting). Output `(BS1.nbas,BS1.nbas,BS2.nbas,3,3)`. See this file's header
comment for the shell-position combinatorics and kernel mapping.
"""
function ∇2ERI_2e3c(BS1::BasisSet, BS2::BasisSet, iA, iB)
    out = zeros(BS1.nbas, BS1.nbas, BS2.nbas, 3, 3)
    return ∇2ERI_2e3c!(out, BS1, BS2, iA, iB)
end

function ∇2ERI_2e3c!(out, BS1::BasisSet, BS2::BasisSet, iA, iB)

    if size(out) != (BS1.nbas, BS1.nbas, BS2.nbas, 3, 3)
        throw(DimensionMismatch("Size of the output array needs to be (N1, N1, N2, 3, 3)."))
    end
    out .= 0.0

    atoms = unique(vcat(BS1.atoms, BS2.atoms))
    basis = vcat(BS1.shells, BS2.shells)
    Bmerged = BasisSet("$(BS1.name*BS2.name)", atoms, basis)

    Aat = BS1.atoms[iA]
    Bat = BS1.atoms[iB]

    Nvals1 = num_basis.(BS1.shells)
    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]

    Nvals2 = num_basis.(BS2.shells)
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]

    for i = 1:BS1.nshells
        atom_i = BS1.shells[i].atom
        for j = i:BS1.nshells
            atom_j = BS1.shells[j].atom
            for k = 1:BS2.nshells
                atom_k = BS2.shells[k].atom

                Xflag = (atom_i == Aat, atom_j == Aat, atom_k == Aat)
                Yflag = (atom_i == Bat, atom_j == Bat, atom_k == Bat)
                (any(Xflag) && any(Yflag)) || continue

                Ni = Nvals1[i]
                Nj = Nvals1[j]
                Nk = Nvals2[k]

                ioff = ao_offset1[i]
                joff = ao_offset1[j]
                koff = ao_offset2[k]

                I = (ioff+1):(ioff+Ni)
                J = (joff+1):(joff+Nj)
                K = (koff+1):(koff+Nk)
                kshell = k + BS1.nshells

                block = zeros(Cdouble, Ni, Nj, Nk, 3, 3)
                for posA in 1:3, posB in 1:3
                    (Xflag[posA] && Yflag[posB]) || continue
                    block .+= posA == posB ?
                        eri3c_hess_same(Bmerged, i, j, kshell, posA) :
                        eri3c_hess_cross(Bmerged, i, j, kshell, posA, posB)
                end

                out[I, J, K, :, :] .+= block
                if i != j
                    out[J, I, K, :, :] .= permutedims(block, (2, 1, 3, 4, 5))
                end
            end
        end
    end

    return out
end
