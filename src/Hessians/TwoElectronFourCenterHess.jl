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

# Scratch-buffer-accepting core, same math as `eri_hess_kernel` above but
# writing the raw libcint call into `buf` and the axis-swapped ((...,6,5))
# result into `dest` (both caller-owned, reused across many calls) instead of
# allocating fresh each time -- see this file's header comment and
# `∇ERI_2e4c!`'s scratch-buffer docstring in `Gradients/TwoElectronGrad.jl`
# for the motivation (profiling found the allocating chain here, up to 16
# calls deep per shell quartet via `∇2ERI_2e4c!`'s posA/posB loop, costing
# several times more per-call allocation than the gradient's already-fixed
# 4-branch case). `dest` must be sized `>= 9*Na*Nb*Nc*Nd`; returns a reshaped
# view into it.
function eri_hess_kernel!(buf::Vector{Cdouble}, dest::Vector{Cdouble}, shls::Vector{Cint}, kern, bset::BasisSet, a, b, c, d)
    Na, Nb, Nc, Nd = num_basis.(bset.basis[[a, b, c, d]])
    n = Na * Nb * Nc * Nd * 9
    shls[1] = a - 1; shls[2] = b - 1; shls[3] = c - 1; shls[4] = d - 1
    lib = bset.lib
    bufv = view(buf, 1:n)
    kern(bufv, shls, lib.atm, lib.natm, lib.bas, lib.nbas, lib.env)
    src = reshape(bufv, Na, Nb, Nc, Nd, 3, 3)
    destv = reshape(view(dest, 1:n), Na, Nb, Nc, Nd, 3, 3)
    permutedims!(destv, src, (1, 2, 3, 4, 6, 5))
    return destv
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

# Scratch-buffer core for `eri_hess_cross` -- `buf`/`t1`/`t2` are caller-owned
# flat `Vector{Cdouble}` scratch (each sized `>= 9*Nmax^4`), `shls` a reused
# `Vector{Cint}` of length 4. Mirrors the allocating version's branch logic
# and permutation specs exactly (mechanical `permutedims`->`permutedims!`
# substitution, verified numerically against it -- see this file's test
# coverage), just tracking which of `t1`/`t2` currently holds the live result
# so the final (posA>posB) swap never targets its own source in place.
function eri_hess_cross!(buf::Vector{Cdouble}, t1::Vector{Cdouble}, t2::Vector{Cdouble}, shls::Vector{Cint},
                          bset::BasisSet, p, q, r, s, posA::Int, posB::Int)
    swp = posA > posB
    a, b = swp ? (posB, posA) : (posA, posB)
    Np, Nq, Nr, Ns = num_basis.(bset.basis[[p, q, r, s]])
    n = Np * Nq * Nr * Ns * 9

    d, dbuf = if (a, b) == (1, 2)
        (eri_hess_kernel!(buf, t1, shls, cint2e_ipvip1_sph!, bset, p, q, r, s), t1)
    elseif (a, b) == (3, 4)
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ipvip1_sph!, bset, r, s, p, q)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (3, 4, 1, 2, 5, 6))
        (dest, t2)
    elseif (a, b) == (1, 3)
        (eri_hess_kernel!(buf, t1, shls, cint2e_ip1ip2_sph!, bset, p, q, r, s), t1)
    elseif (a, b) == (1, 4)
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ip1ip2_sph!, bset, p, q, s, r)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (1, 2, 4, 3, 5, 6))
        (dest, t2)
    elseif (a, b) == (2, 3)
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ip1ip2_sph!, bset, q, p, r, s)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (2, 1, 3, 4, 5, 6))
        (dest, t2)
    elseif (a, b) == (2, 4)
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ip1ip2_sph!, bset, q, p, s, r)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (2, 1, 4, 3, 5, 6))
        (dest, t2)
    else
        throw(ArgumentError("unexpected position pair ($a,$b)"))
    end

    if swp
        other = dbuf === t1 ? t2 : t1
        dest = reshape(view(other, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, d, (1, 2, 3, 4, 6, 5))
        return dest
    end
    return d
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

# Scratch-buffer core for `eri_hess_same`, same relationship to it as
# `eri_hess_cross!` has to `eri_hess_cross`.
function eri_hess_same!(buf::Vector{Cdouble}, t1::Vector{Cdouble}, t2::Vector{Cdouble}, shls::Vector{Cint},
                         bset::BasisSet, p, q, r, s, posK::Int)
    Np, Nq, Nr, Ns = num_basis.(bset.basis[[p, q, r, s]])
    n = Np * Nq * Nr * Ns * 9
    if posK == 1
        return eri_hess_kernel!(buf, t1, shls, cint2e_ipip1_sph!, bset, p, q, r, s)
    elseif posK == 2
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ipip1_sph!, bset, q, p, r, s)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (2, 1, 3, 4, 5, 6))
        return dest
    elseif posK == 3
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ipip1_sph!, bset, r, s, p, q)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (3, 4, 1, 2, 5, 6))
        return dest
    elseif posK == 4
        raw = eri_hess_kernel!(buf, t1, shls, cint2e_ipip1_sph!, bset, s, r, p, q)
        dest = reshape(view(t2, 1:n), Np, Nq, Nr, Ns, 3, 3)
        permutedims!(dest, raw, (3, 4, 2, 1, 5, 6))
        return dest
    else
        throw(ArgumentError("posK must be 1..4"))
    end
end

"""
    ∇2ERI_2e4c(BS::BasisSet, iA::Int, iB::Int, i::Int, j::Int, k::Int, l::Int)
    ∇2ERI_2e4c(BS::BasisSet, Xflag::NTuple{4,Bool}, Yflag::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int)

Shell-quartet-level dense 4-center ERI Hessian: `∂²(ij|kl)/∂R_iA∂R_iB` for
shells `i,j,k,l` w.r.t. atoms `iA,iB`'s three Cartesian directions each,
with `R_iA`/`R_iB` in bohr (see [Hessians](@ref) for units), as an
`(Ni,Nj,Nk,Nl,3,3)` block. No Schwarz screening either way -- callers
wanting that should screen before calling.

Two forms, same relationship as `∇ERI_2e4c`'s: the `iA::Int,iB::Int` form
computes `Xflag`/`Yflag` (which of `i,j,k,l` sit on `iA`/`iB`) and returns
the free zero (no libcint call) whenever no shell touches `iA`, no shell
touches `iB`, or `iA==iB` with all four shells on that one atom
(translational invariance). The `Xflag,Yflag::NTuple{4,Bool}` form is the
unchecked core: no membership computation, no zero-skip -- for callers
that have already screened at a higher level and precomputed the flags
once outside a hot loop.
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
    Nijkl = Ni * Nj * Nk * Nl
    buf = zeros(Cdouble, 9 * Nijkl)
    t1 = zeros(Cdouble, 9 * Nijkl)
    t2 = zeros(Cdouble, 9 * Nijkl)
    shls = zeros(Cint, 4)
    return ∇2ERI_2e4c!(out, BS, Xflag, Yflag, i, j, k, l, buf, t1, t2, shls)
end

"""
    ∇2ERI_2e4c!(out, BS::BasisSet, Xflag, Yflag, i, j, k, l, buf, t1, t2, shls)

Scratch-buffer-accepting core: `buf`/`t1`/`t2` (each `>= 9*Nmax^4`, `Nmax`
= the largest `num_basis` over any shell the caller will ever pass) and
`shls` (length 4) are caller-owned and reused across every call instead of
allocated fresh. Not thread-safe to share: each concurrent caller (e.g.
each worker task) needs its own scratch buffers.
"""
function ∇2ERI_2e4c!(out, BS::BasisSet, Xflag::NTuple{4,Bool}, Yflag::NTuple{4,Bool}, i::Int, j::Int, k::Int, l::Int,
                      buf::Vector{Cdouble}, t1::Vector{Cdouble}, t2::Vector{Cdouble}, shls::Vector{Cint})
    # This loop calls eri_hess_same!/eri_hess_cross! up to 16 times per
    # shell quartet (4x4 posA/posB combinations, vs. the gradient's 4
    # branches), so the old allocating path cost proportionally more GC
    # churn per call -- see Gradients/TwoElectronGrad.jl's ∇ERI_2e4c! for
    # the analogous gradient-side fix.
    Ni, Nj, Nk, Nl = num_basis.(BS.basis[[i, j, k, l]])
    if size(out) != (Ni, Nj, Nk, Nl, 3, 3)
        throw(DimensionMismatch("Size of the output array needs to be ($Ni, $Nj, $Nk, $Nl, 3, 3)."))
    end

    out .= 0.0

    for posA in 1:4
        Xflag[posA] || continue
        for posB in 1:4
            Yflag[posB] || continue
            d = posA == posB ? eri_hess_same!(buf, t1, t2, shls, BS, i, j, k, l, posA) :
                                eri_hess_cross!(buf, t1, t2, shls, BS, i, j, k, l, posA, posB)
            out .+= d
        end
    end

    return out
end
