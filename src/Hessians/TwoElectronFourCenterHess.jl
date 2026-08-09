# --- Shell-quartet-level dense 4-center ERI Hessian ---
#
# GaussianBasis-idiomatic light bridge to libcint's second-derivative 2e
# kernels, mirroring the shell-quartet gradient primitive in
# `Gradients/TwoElectronGrad.jl` one derivative order up: no Schwarz
# screening here (that's Fermi's own loop's job, same split the gradient
# primitive already uses -- see that file's header comment), only the free,
# exact (not threshold-based) structural zero checks. Everything below this
# comment (`eri_hess_kernel`/`eri_hess_same`/`eri_hess_cross`) is ported
# near-verbatim from Fermi.jl's `Hessians/TwoElectronHess.jl`, which already
# did the hard, error-prone work of confirming libcint's kernel-to-shell-pair
# mapping and axis orientation against `cint_funcs.h` and finite difference --
# moved here rather than re-derived, and no longer needed at the Fermi.jl
# call site once Fermi switches to calling this instead.
#
# Which kernel gives which shell-pair's second derivative is NOT what the
# "ip1", "ip2" naming suggests at first glance:
#
#   cint2e_ipip1  "(NABLA NABLA i j|R12|k l)"   both derivatives on shell i
#                                                (same shell, position 1)
#   cint2e_ipvip1 "(NABLA i NABLA j|R12|k l)"    one derivative each on shells
#                                                i,j (SAME electron pair --
#                                                i,j both "bra", positions 1,2)
#   cint2e_ip1ip2 "(NABLA i j|R12|NABLA k l)"    one derivative each on shells
#                                                i,k (OPPOSITE electron pairs
#                                                -- i is "bra" position 1, k
#                                                is "ket" position 3)
#
# ip1ip2 giving the opposite-pair (i.e. i vs k, not i vs j) cross derivative
# directly is the crucial piece -- with it, all 10 distinct shell-pair second
# derivatives of a quartet (same-shell x4, same-pair-cross x2, opposite-pair-
# cross x4) are directly available via these 3 kernels plus shell-argument
# reordering (exploiting the standard 8-fold ERI permutation symmetry
# (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq)=...).
#
# Every raw kernel whose two derivatives land on DIFFERENT shells (ipvip1,
# ip1ip2) has its two derivative-component axes in (position-2-or-4,
# position-1-or-3) order in the raw buffer -- reversed from the naive
# expectation, same pattern found for the one-electron mixed kernels
# (cint1e_iprinvip_sph!). ipip1 (same shell, mixed partials of the same
# point commute) needs no such correction.

function eri_hess_kernel(kern, bset::BasisSet, a, b, c, d)
    Na, Nb, Nc, Nd = num_basis.(bset.basis[[a, b, c, d]])
    buf = zeros(Cdouble, 9 * Na * Nb * Nc * Nd)
    kern(buf, [a, b, c, d], bset.lib)
    return permutedims(reshape(buf, Na, Nb, Nc, Nd, 3, 3), (1, 2, 3, 4, 6, 5))
end

# Second derivative of the (p,q,r,s) integral w.r.t. the shells at argument
# positions posA, posB (1..4, corresponding to p,q,r,s respectively), returned
# as a (Np,Nq,Nr,Ns,3,3) tensor in (p,q,r,s) AO-axis order regardless of which
# positions were differentiated. posA == posB (same-shell case) is handled by
# eri_hess_same below; this function only handles posA != posB.
function eri_hess_cross(bset::BasisSet, p, q, r, s, posA::Int, posB::Int)
    swp = posA > posB
    a, b = swp ? (posB, posA) : (posA, posB)

    d = if (a, b) == (1, 2)
        eri_hess_kernel(cint2e_ipvip1_sph!, bset, p, q, r, s)
    elseif (a, b) == (3, 4)
        raw = eri_hess_kernel(cint2e_ipvip1_sph!, bset, r, s, p, q)
        permutedims(raw, (3, 4, 1, 2, 5, 6))
    elseif (a, b) == (1, 3)
        eri_hess_kernel(cint2e_ip1ip2_sph!, bset, p, q, r, s)
    elseif (a, b) == (1, 4)
        raw = eri_hess_kernel(cint2e_ip1ip2_sph!, bset, p, q, s, r)
        permutedims(raw, (1, 2, 4, 3, 5, 6))
    elseif (a, b) == (2, 3)
        raw = eri_hess_kernel(cint2e_ip1ip2_sph!, bset, q, p, r, s)
        permutedims(raw, (2, 1, 3, 4, 5, 6))
    elseif (a, b) == (2, 4)
        raw = eri_hess_kernel(cint2e_ip1ip2_sph!, bset, q, p, s, r)
        permutedims(raw, (2, 1, 4, 3, 5, 6))
    else
        throw(ArgumentError("unexpected position pair ($a,$b)"))
    end

    # d(posB,posA)[...,x,y] = d(posA,posB)[...,y,x] (mixed partials commute)
    return swp ? permutedims(d, (1, 2, 3, 4, 6, 5)) : d
end

# Same-shell second derivative: both derivatives land on the shell at position
# posK (1..4). Returned in (p,q,r,s) AO-axis order.
function eri_hess_same(bset::BasisSet, p, q, r, s, posK::Int)
    if posK == 1
        eri_hess_kernel(cint2e_ipip1_sph!, bset, p, q, r, s)
    elseif posK == 2
        raw = eri_hess_kernel(cint2e_ipip1_sph!, bset, q, p, r, s)
        permutedims(raw, (2, 1, 3, 4, 5, 6))
    elseif posK == 3
        raw = eri_hess_kernel(cint2e_ipip1_sph!, bset, r, s, p, q)
        permutedims(raw, (3, 4, 1, 2, 5, 6))
    elseif posK == 4
        raw = eri_hess_kernel(cint2e_ipip1_sph!, bset, s, r, p, q)
        permutedims(raw, (3, 4, 2, 1, 5, 6))
    else
        throw(ArgumentError("posK must be 1..4"))
    end
end

"""
    ∇2ERI_2e4c(BS::BasisSet, iA::Int, iB::Int, i::Int, j::Int, k::Int, l::Int)
    ∇2ERI_2e4c(BS::BasisSet, Xflag::NTuple{4,Bool}, Yflag::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)

Shell-quartet-level dense 4-center ERI Hessian: `∂²(ij|kl)/∂R_iA∂R_iB` for
shells `i,j,k,l` w.r.t. atoms `iA,iB`'s three Cartesian directions each, as
an `(Ni,Nj,Nk,Nl,3,3)` block. No Schwarz screening either way -- see this
file's header comment; callers wanting that should screen before calling,
same as `∇ERI_2e4c(BS,iA,i,j,k,l)`'s callers already do.

Two forms, same relationship as `∇ERI_2e4c`'s: the `iA::Int,iB::Int` form
computes `Xflag`/`Yflag` (which of `i,j,k,l` sit on `iA`/`iB`, via `===` --
see `on_atom_flags`) and returns the free zero (no libcint call) whenever no
shell touches `iA`, no shell touches `iB`, or `iA==iB` with ALL FOUR shells
on that one atom (translational invariance -- the integral is then a
function of the shells' *relative* positions only, all identically zero
when every shell shares one center, so it doesn't depend on that atom's
position at all, and neither does any derivative of it). The
`Xflag::NTuple{4,Bool},Yflag::NTuple{4,Bool}` form is the unchecked core:
no membership computation, no zero-skip (not even the `iA==iB` translational
-invariance case, since this form doesn't receive `iA`/`iB` at all to check
it) -- for callers that have already screened at a higher level and
precomputed the flags once outside a hot loop.
"""
function ∇2ERI_2e4c(BS::BasisSet, iA::Int, iB::Int, i::Int, j::Int, k::Int, l::Int)
    Xflag = on_atom_flags(BS, iA, i, j, k, l)
    Yflag = on_atom_flags(BS, iB, i, j, k, l)
    Ni, Nj, Nk, Nl = num_basis.(BS.basis[[i, j, k, l]])
    out = zeros(Ni, Nj, Nk, Nl, 3, 3)
    (!any(Xflag) || !any(Yflag) || (iA == iB && all(Xflag))) && return out
    return ∇2ERI_2e4c!(out, BS, Xflag, Yflag, i, j, k, l)
end

function ∇2ERI_2e4c!(out, BS::BasisSet, iA::Int, iB::Int, i::Int, j::Int, k::Int, l::Int)
    Xflag = on_atom_flags(BS, iA, i, j, k, l)
    Yflag = on_atom_flags(BS, iB, i, j, k, l)
    if !any(Xflag) || !any(Yflag) || (iA == iB && all(Xflag))
        out .= 0.0
        return out
    end
    return ∇2ERI_2e4c!(out, BS, Xflag, Yflag, i, j, k, l)
end

function ∇2ERI_2e4c(BS::BasisSet, Xflag::NTuple{4,Bool}, Yflag::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)
    Ni, Nj, Nk, Nl = num_basis.(BS.basis[[i, j, k, l]])
    out = zeros(Ni, Nj, Nk, Nl, 3, 3)
    return ∇2ERI_2e4c!(out, BS, Xflag, Yflag, i, j, k, l)
end

function ∇2ERI_2e4c!(out, BS::BasisSet, Xflag::NTuple{4,Bool}, Yflag::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)
    Ni, Nj, Nk, Nl = num_basis.(BS.basis[[i, j, k, l]])
    if size(out) != (Ni, Nj, Nk, Nl, 3, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Ni, $Nj, $Nk, $Nl, 3, 3)."))
    end

    out .= 0.0

    for posA in 1:4
        Xflag[posA] || continue
        for posB in 1:4
            Yflag[posB] || continue
            d = posA == posB ? eri_hess_same(BS, i, j, k, l, posA) :
                                eri_hess_cross(BS, i, j, k, l, posA, posB)
            out .+= d
        end
    end

    return out
end
