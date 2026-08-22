# Mutating, shell-pair-level backend for ERI_2e2c(BS): writes into a
# caller-supplied `out` instead of allocating. Dispatches on the integral
# backend, same ghost-basis-function trick as ERI_2e3c! for the ACSint
# fallback (here standing in for the two missing centers of a 4-center
# integral).
function ERI_2e2c!(out, BS::BasisSet, i, j)
    generate_ERI_quartet!(out, BS.shells[i], _ghostBF, BS.shells[j], _ghostBF)
end

function ERI_2e2c!(out, BS::BasisSet{LCint}, i, j)
    cint2c2e_sph!(out, [i,j], BS.lib)
end

"""
    ERI_2e2c(BS::BasisSet) -> Matrix{Float64}

Compute the full two-electron two-center integral matrix `(P|Q)` for `BS`
(the auxiliary/fitting basis) -- the Coulomb metric `J_{PQ}` used in
density fitting / resolution-of-the-identity approximations. Returns a
dense, symmetric `nbas × nbas` matrix. For repeated calls, see `ERI_2e2c!`.
"""
function ERI_2e2c(BS::BasisSet)
    out = zeros(BS.nbas, BS.nbas)
    ERI_2e2c!(out, BS)
end

function ERI_2e2c!(out, BS::BasisSet)
    get_1e_matrix!(ERI_2e2c!, out, BS)
end