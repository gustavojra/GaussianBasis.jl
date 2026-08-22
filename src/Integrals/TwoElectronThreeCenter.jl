const _ghostBF = CartesianShell(0, [1.0], [0.0], Atom(1, 1.0, [0.0, 0.0, 0.0]))

# Mutating, shell-triple-level backend for ERI_2e3c(BS1, BS2): writes into a
# caller-supplied `out` instead of allocating. Dispatches on the integral
# backend: LCint uses libcint's native 3-center kernel directly; the ACSint
# fallback below instead evaluates it as a 4-center integral against a ghost
# (zero-charge, s-type) basis function standing in for the missing 4th center.
function ERI_2e3c!(out, BS::BasisSet{LCint}, i, j, k)
    cint3c2e_sph!(out, [i,j,k], BS.lib)
end

function ERI_2e3c!(out, BS1::BasisSet, BS2::BasisSet, i, j, k)
    generate_ERI_quartet!(out, BS1.shells[i], BS1.shells[j], BS2.shells[k], _ghostBF)
end

"""
    ERI_2e3c(BS1::BasisSet, BS2::BasisSet) -> Array{Float64,3}

Compute the full two-electron three-center integral tensor `(μν|P)`, with
`μ,ν` running over `BS1`'s AOs (the "regular" orbital basis) and `P` over
`BS2`'s (the auxiliary/fitting basis) -- the building block for density
fitting / resolution-of-the-identity approximations. Returns a dense
`BS1.nbas × BS1.nbas × BS2.nbas` array, symmetric under `μ↔ν` swap. For
repeated calls, see `ERI_2e3c!`.
"""
function ERI_2e3c(BS1::BasisSet, BS2::BasisSet)
    out = zeros(BS1.nbas, BS1.nbas, BS2.nbas)
    ERI_2e3c!(out, BS1, BS2)
end

function ERI_2e3c!(out, BS1::BasisSet, BS2::BasisSet)

    # Pre compute number of basis per shell
    Nvals1 = num_basis.(BS1.shells)
    Nvals2 = num_basis.(BS2.shells)
    Nmax1 = maximum(Nvals1)
    Nmax2 = maximum(Nvals2)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]

    allocate(body) = body(zeros(Cdouble, Nmax1^2*Nmax2))
    workerpool(allocate, 1:BS2.nshells; chunksize=1) do k, buf
        @inbounds begin
            Nk = Nvals2[k]
            koff = ao_offset2[k]
            for i in 1:BS1.nshells
                Ni = Nvals1[i]
                ioff = ao_offset1[i]
                for j in i:BS1.nshells
                    Nj = Nvals1[j]
                    joff = ao_offset1[j]

                    # Call libcint
                    ERI_2e3c!(buf, BS1, BS2, i, j, k)

                    # Loop through shell block and save unique elements
                    for ks = 1:Nk
                        K = koff + ks
                        for js = 1:Nj
                            J = joff + js
                            for is = 1:Ni
                                I = ioff + is
                                J < I ? break : nothing
                                out[I,J,K] = buf[is + Ni*(js-1) + Ni*Nj*(ks-1)]
                                out[J,I,K] = out[I,J,K]
                            end
                        end
                    end
                end
            end
        end #inbounds
    end
    return out
end

function ERI_2e3c!(out, BS1::BasisSet{LCint}, BS2::BasisSet{LCint})

    atoms = unique(vcat(BS1.atoms, BS2.atoms))
    basis = vcat(BS1.shells, BS2.shells)

    Bmerged = BasisSet("$(BS1.name*BS2.name)", atoms, basis)

    # Pre compute number of basis per shell
    Nvals1 = num_basis.(BS1.shells)
    Nvals2 = num_basis.(BS2.shells)
    Nmax1 = maximum(Nvals1)
    Nmax2 = maximum(Nvals2)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]


    allocate(body) = body(zeros(Cdouble, Nmax1^2*Nmax2))
    workerpool(allocate, 1:BS2.nshells; chunksize = 1) do k, buf
        @inbounds begin
            Nk = Nvals2[k]
            koff = ao_offset2[k]
            for i in 1:BS1.nshells
                Ni = Nvals1[i]
                ioff = ao_offset1[i]
                for j in i:BS1.nshells
                    Nj = Nvals1[j]
                    joff = ao_offset1[j]

                    # Call libcint
                    ERI_2e3c!(buf, Bmerged, i, j, k+BS1.nshells)

                    # Loop through shell block and save unique elements
                    for ks = 1:Nk
                        K = koff + ks
                        for js = 1:Nj
                            J = joff + js
                            for is = 1:Ni
                                I = ioff + is
                                J < I ? break : nothing
                                out[I,J,K] = buf[is + Ni*(js-1) + Ni*Nj*(ks-1)]
                                out[J,I,K] = out[I,J,K]
                            end
                        end
                    end
                end
            end
        end #inbounds
    end
    return out
end
