# Mutating, shell-pair-level forms of overlap/kinetic/nuclear(BS, i, j):
# write into a caller-supplied `out` (sized (Ni,Nj)) instead of allocating.
# Dispatches on the integral backend (LCint vs. the ACSint fallback below).
# Backend: Libcint
function overlap!(out, BS::BasisSet{LCint}, i, j)
    cint1e_ovlp_sph!(out, @SVector([i,j]), BS.lib)
end

function kinetic!(out, BS::BasisSet{LCint}, i, j)
    cint1e_kin_sph!(out, @SVector([i,j]), BS.lib)
end

function nuclear!(out, BS::BasisSet{LCint}, i, j)
    cint1e_nuc_sph!(out, @SVector([i,j]), BS.lib)
end

# Backend: ACSint -- fall back 
function overlap!(out, BS::BasisSet, i, j)
    generate_S_pair!(out, BS, i ,j)
end

function kinetic!(out, BS::BasisSet, i, j)
    generate_T_pair!(out, BS, i ,j)
end

function nuclear!(out, BS::BasisSet, i, j)
    generate_V_pair!(out, BS, i ,j)
end

# General

"""
    overlap(BS::BasisSet, i, j) -> Matrix{Float64}

Compute the AO overlap block `⟨i|j⟩` for shells `i` and `j` of `BS` (shell
indices, not AO indices), returned as an `(Ni,Nj)` matrix where `Ni`/`Nj`
are the number of basis functions in each shell. For the full overlap
matrix, see [`overlap(BS)`](@ref overlap(::BasisSet)).
"""
function overlap(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]))
    overlap!(out, BS, i, j)
    return out
end

"""
    kinetic(BS::BasisSet, i, j) -> Matrix{Float64}

Compute the AO kinetic energy block for shells `i` and `j` of `BS` (shell
indices). Returned as an `(Ni,Nj)` matrix. For the full kinetic energy
matrix, see [`kinetic(BS)`](@ref kinetic(::BasisSet)).
"""
function kinetic(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]))
    kinetic!(out, BS, i, j)
    return out
end

"""
    nuclear(BS::BasisSet, i, j) -> Matrix{Float64}

Compute the AO nuclear attraction block (summed over every nucleus in
`BS.atoms`) for shells `i` and `j` of `BS` (shell indices). Returned as an
`(Ni,Nj)` matrix. For the full nuclear attraction matrix, see
[`nuclear(BS)`](@ref nuclear(::BasisSet)).
"""
function nuclear(BS::BasisSet, i, j)
    out = zeros(num_basis(BS.basis[i]), num_basis(BS.basis[j]))
    nuclear!(out, BS, i, j)
    return out
end

"""
    overlap(BS::BasisSet) -> Matrix{Float64}

Compute the AO overlap matrix `S` for `BS`. Returns a dense, symmetric
`nbas × nbas` matrix. For repeated calls (e.g. reusing a preallocated
array), see `overlap!`.
"""
overlap(BS::BasisSet) = get_1e_matrix(overlap!, BS)

"""
    kinetic(BS::BasisSet) -> Matrix{Float64}

Compute the AO kinetic energy matrix `T` for `BS`. Returns a dense,
symmetric `nbas × nbas` matrix. For repeated calls, see `kinetic!`.
"""
kinetic(BS::BasisSet) = get_1e_matrix(kinetic!, BS)

"""
    nuclear(BS::BasisSet) -> Matrix{Float64}

Compute the AO nuclear attraction matrix `V` for `BS` (potential from every
nucleus in `BS.atoms`, summed). Returns a dense, symmetric `nbas × nbas`
matrix. For repeated calls, see `nuclear!`.
"""
nuclear(BS::BasisSet) = get_1e_matrix(nuclear!, BS)

# `overlap!`/`kinetic!`/`nuclear!(out, BS)` write into a caller-supplied
# `nbas × nbas` `out` instead of allocating -- use these in a hot loop (e.g.
# repeated calls across a geometry scan) to avoid reallocating every time.
overlap!(out, BS::BasisSet) = get_1e_matrix!(overlap!, out, BS)
kinetic!(out, BS::BasisSet) = get_1e_matrix!(kinetic!, out, BS)
nuclear!(out, BS::BasisSet) = get_1e_matrix!(nuclear!, out, BS)

function get_1e_matrix(callback, BS::BasisSet, rank::Integer = 0)
    out = zeros(eltype(BS.atoms[1].xyz), BS.nbas, BS.nbas, Iterators.repeated(3, rank)...)
    return get_1e_matrix!(callback, out, BS, rank)
end

function get_1e_matrix!(callback, out, BS::BasisSet, rank::Integer = 0)
    Nvals = num_basis.(BS.basis)
    Nmax = maximum(Nvals)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]

    ijs = Iterators.flatten((((i,j) for i = 1:j) for j = 1:BS.nshells))

    allocate(body) = body(zeros(Cdouble, 3^rank * Nmax^2))
    workerpool(allocate, ijs; chunksize=10) do (i,j), buf
            Ni = Nvals[i]
            Nj = Nvals[j]
            Nij = Ni*Nj
            ioff = ao_offset[i]
            joff = ao_offset[j]

            callback(buf, BS, i, j)
            I = (ioff+1):(ioff+Ni)
            J = (joff+1):(joff+Nj)

            # Get strides for each cartesian product
            for (n, ks) in enumerate(Iterators.product(Iterators.repeated(1:3, rank)...))
                r = Nij*(n-1)+1:n*Nij
                kvals = reshape(view(buf, r), Ni, Nj)
                out[I,J,ks...] .+= kvals
                if i != j
                    out[J,I,ks...] .+= kvals'
                end
            end
    end
    return out
end

########################################################### 
########################################################### 
#                       Mixed basis                       #
########################################################### 
########################################################### 

# Mutating, shell-pair-level forms of the mixed-basis overlap/kinetic/nuclear
# above: write into a caller-supplied `out` instead of allocating.
# Backend: ACSint -- fall back
function overlap!(out, BS1::BasisSet, BS2::BasisSet, i, j)
    generate_S_pair!(out, BS1.basis[i], BS2.basis[j])
end

function kinetic!(out, BS1::BasisSet, BS2::BasisSet, i, j)
    generate_T_pair!(out, BS1.basis[i], BS2.basis[j])
end

function nuclear!(out, BS1::BasisSet, BS2::BasisSet, i, j)
    generate_V_pair!(out, BS1.basis[i], BS2.basis[j], BS1.atoms)
end

# General

"""
    overlap(BS1::BasisSet, BS2::BasisSet, i, j) -> Matrix{Float64}

Mixed-basis overlap block `⟨i|j⟩` between shell `i` of `BS1` and shell `j`
of `BS2` (shell indices into each respective basis set). Returned as an
`(Ni,Nj)` matrix. For the full mixed-basis matrix, see
[`overlap(BS1, BS2)`](@ref overlap(::BasisSet, ::BasisSet)).
"""
function overlap(BS1::BasisSet, BS2::BasisSet, i, j)
    out = zeros(num_basis(BS1.basis[i]), num_basis(BS2.basis[j]))
    overlap!(out, BS1, BS2, i, j)
    return out
end

"""
    kinetic(BS1::BasisSet, BS2::BasisSet, i, j) -> Matrix{Float64}

Mixed-basis kinetic energy block between shell `i` of `BS1` and shell `j`
of `BS2`. Returned as an `(Ni,Nj)` matrix. For the full mixed-basis matrix,
see [`kinetic(BS1, BS2)`](@ref kinetic(::BasisSet, ::BasisSet)).
"""
function kinetic(BS1::BasisSet, BS2::BasisSet, i, j)
    out = zeros(num_basis(BS1.basis[i]), num_basis(BS2.basis[j]))
    kinetic!(out, BS1, BS2, i, j)
    return out
end

"""
    nuclear(BS1::BasisSet, BS2::BasisSet, i, j) -> Matrix{Float64}

Mixed-basis nuclear attraction block between shell `i` of `BS1` and shell
`j` of `BS2`, using the nuclei of `BS1`. Returned as an `(Ni,Nj)` matrix.
For the full mixed-basis matrix, see
[`nuclear(BS1, BS2)`](@ref nuclear(::BasisSet, ::BasisSet)).
"""
function nuclear(BS1::BasisSet, BS2::BasisSet, i, j)
    out = zeros(num_basis(BS1.basis[i]), num_basis(BS2.basis[j]))
    nuclear!(out, BS1, BS2, i, j)
    return out
end

"""
    overlap(BS1::BasisSet, BS2::BasisSet) -> Matrix{Float64}

Mixed-basis overlap matrix `S_{μν} = ⟨μ|ν⟩` with `μ` running over `BS1`'s
AOs and `ν` over `BS2`'s. Returns a dense `BS1.nbas × BS2.nbas` matrix (not
generally symmetric, and not square unless `BS1`/`BS2` have equal size).
Useful e.g. for projecting a density or orbitals expressed in one basis
onto another. For repeated calls, see `overlap!`.
"""
overlap(BS1::BasisSet, BS2::BasisSet) = get_1e_matrix(overlap!, BS1, BS2)

"""
    kinetic(BS1::BasisSet, BS2::BasisSet) -> Matrix{Float64}

Mixed-basis kinetic energy matrix, `BS1.nbas × BS2.nbas`. See
[`overlap(BS1, BS2)`](@ref overlap(::BasisSet, ::BasisSet)) for the general
mixed-basis convention. For repeated calls, see `kinetic!`.
"""
kinetic(BS1::BasisSet, BS2::BasisSet) = get_1e_matrix(kinetic!, BS1, BS2)

"""
    nuclear(BS1::BasisSet, BS2::BasisSet) -> Matrix{Float64}

Mixed-basis nuclear attraction matrix (nuclei taken from `BS1.atoms`),
`BS1.nbas × BS2.nbas`. See
[`overlap(BS1, BS2)`](@ref overlap(::BasisSet, ::BasisSet)) for the general
mixed-basis convention. For repeated calls, see `nuclear!`.
"""
nuclear(BS1::BasisSet, BS2::BasisSet) = get_1e_matrix(nuclear!, BS1, BS2)

# `overlap!`/`kinetic!`/`nuclear!(out, BS1, BS2)` write into a caller-supplied
# `BS1.nbas × BS2.nbas` `out` instead of allocating.
overlap!(out, BS1::BasisSet, BS2::BasisSet) = get_1e_matrix!(overlap!, out, BS1, BS2)
kinetic!(out, BS1::BasisSet, BS2::BasisSet) = get_1e_matrix!(kinetic!, out, BS1, BS2)
nuclear!(out, BS1::BasisSet, BS2::BasisSet) = get_1e_matrix!(nuclear!, out, BS1, BS2)

function get_1e_matrix(callback, BS1::BasisSet, BS2::BasisSet)
    out = zeros(BS1.nbas, BS2.nbas)
    get_1e_matrix!(callback, out, BS1, BS2)
end

function get_1e_matrix!(callback, out, BS1::BasisSet, BS2::BasisSet)

    # Pre compute number of basis per shell
    Nvals1 = num_basis.(BS1.basis)
    Nvals2 = num_basis.(BS2.basis)
    Nmax1 = maximum(Nvals1)
    Nmax2 = maximum(Nvals2)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]

    allocate(body) = body(zeros(Cdouble, Nmax1*Nmax2))
    workerpool(allocate, 1:BS1.nshells; chunksize = 1) do i, buf
        @inbounds begin
            Li = Nvals1[i]
            buf = zeros(Cdouble, Nmax1*Nmax2)
            ioff = ao_offset1[i]
            for j in 1:BS2.nshells
                Lj = Nvals2[j]
                joff = ao_offset2[j]

                # Call libcint
                callback(buf, BS1, BS2, i, j)

                # Loop through shell block and save unique elements
                for js = 1:Lj
                    J = joff + js
                    for is = 1:Li
                        I = ioff + is
                        out[I,J] = buf[is + Li*(js-1)]
                    end
                end
            end
        end #inbounds
    end
    return out
end

function get_1e_matrix!(callback, out, BS1::BasisSet{LCint}, BS2::BasisSet{LCint})

    atoms = unique(vcat(BS1.atoms, BS2.atoms))
    basis = vcat(BS1.basis, BS2.basis)

    Bmerged = BasisSet("$(BS1.name*BS2.name)", atoms, basis)

    # Pre compute number of basis per shell
    Nvals1 = num_basis.(BS1.basis)
    Nvals2 = num_basis.(BS2.basis)
    Nmax1 = maximum(Nvals1)
    Nmax2 = maximum(Nvals2)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]

    allocate(body) = body(zeros(Cdouble, Nmax1*Nmax2))
    workerpool(allocate, 1:BS1.nshells; chunksize = 1) do i, buf
        @inbounds begin
            Li = Nvals1[i]
            ioff = ao_offset1[i]
            for j in 1:BS2.nshells
                Lj = Nvals2[j]
                joff = ao_offset2[j]

                # Call libcint
                callback(buf, Bmerged, i, j+BS1.nshells)

                # Loop through shell block and save unique elements
                for js = 1:Lj
                    J = joff + js
                    for is = 1:Li
                        I = ioff + is
                        out[I,J] = buf[is + Li*(js-1)]
                    end
                end
            end
        end #inbounds
    end
    return out
end
