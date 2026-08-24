# Test if elements in the sparse array (S) are present
# in the dense array (D)
function SinD(S, D, verbose = false)
    Si = S[1]
    Sx = S[2]
    Sy = S[3]
    Sz = S[4]

    Dxs = D[:,:,:,:,1]
    Dys = D[:,:,:,:,2]
    Dzs = D[:,:,:,:,3]
    for i in eachindex(Si)
        idx = Si[i]
        x = Sx[i]
        y = Sy[i]
        z = Sz[i]
        Dx = Dxs[idx...]
        Dy = Dys[idx...]
        Dz = Dzs[idx...]

        for (k,Dk) in [(x,Dx), (y,Dy), (z,Dz)]
            if abs(k - Dk) > 1e-10
                println("$idx diff found: $k against $Dk")
                return false
            end
        end
        verbose ? println("$idx OK") : nothing
    end
    return true
end

atoms = Molecules.parse_string("""
C   -2.131551124300    2.286168823700    0.000000000000
H   -1.061551124300    2.286168823700    0.000000000000
H   -2.488213906200    1.408104616400    0.496683911300
H   -2.488218762100    2.295059432700   -1.008766153900
H   -2.488220057000    3.155340844300    0.512081313000""")

bs = BasisSet("cc-pvdz", atoms)

@testset "∂S/∂X" begin
    for iA = 1:length(atoms)
        dS = ∇overlap(bs, iA)
        for k = 1:3
            @test dS[:,:,k] ≈ ∇FD_overlap(bs, iA, k)
        end
    end
end

@testset "∂T/∂X" begin
    for iA = 1:length(atoms)
        dT = ∇kinetic(bs, iA)
        for k = 1:3
            @test dT[:,:,k] ≈ ∇FD_kinetic(bs, iA, k)
        end
    end
end

@testset "∂V/∂X" begin
    for iA = 1:length(atoms)
        dV = ∇nuclear(bs, iA)
        for k = 1:3
            @test dV[:,:,k] ≈ ∇FD_nuclear(bs, iA, k)
        end
    end
end

@testset "Shell-pair-level gradients" begin
    # ∇overlap/∇kinetic/∇nuclear(BS, iA, i, j) -- assembling every shell
    # pair's (Ni,Nj,3) block should reproduce the already-validated
    # whole-array ∇overlap/∇kinetic/∇nuclear(BS, iA) exactly (same
    # underlying computation, no independent derivation, just a different
    # slicing -- so this checks self-consistency, not the integrals
    # themselves, which the FD tests above already cover).
    Nvals = GaussianBasis.num_basis.(bs.shells)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:bs.nshells]

    function assemble(f, iA)
        out = zeros(bs.nbas, bs.nbas, 3)
        for i in 1:bs.nshells, j in 1:bs.nshells
            Ni, Nj = Nvals[i], Nvals[j]
            I = (ao_offset[i]+1):(ao_offset[i]+Ni)
            J = (ao_offset[j]+1):(ao_offset[j]+Nj)
            out[I,J,:] .= f(bs, iA, i, j)
        end
        return out
    end

    for iA = 1:length(atoms)
        @test ∇overlap(bs, iA) ≈ assemble(∇overlap, iA) atol=1e-12
        @test ∇kinetic(bs, iA) ≈ assemble(∇kinetic, iA) atol=1e-12
        @test ∇nuclear(bs, iA) ≈ assemble(∇nuclear, iA) atol=1e-12
    end
end

@testset "∂[ij|kl]/∂X" begin
    @testset "Dense" begin
        for iA = 1:length(atoms)
            dERI = ∇ERI_2e4c(bs, iA)
            for k = 1:3
                @test dERI[:,:,:,:,k] ≈ ∇FD_ERI_2e4c(bs, iA, k)
            end
        end
    end

    @testset "Sparse" begin
        for iA = 1:length(atoms)
            sparse = ∇sparseERI_2e4c(bs, iA)
            dense = ∇ERI_2e4c(bs, iA)
            @test SinD(sparse, dense)
        end
    end

    @testset "Shell-quartet-level" begin
        # ∇ERI_2e4c(BS, iA, i, j, k, l) -- assembling every shell quartet's
        # (Ni,Nj,Nk,Nl,3) block should reproduce the already-validated dense
        # ∇ERI_2e4c(BS, iA) exactly (same underlying formula, no
        # permutation-symmetry propagation, just direct per-quartet calls --
        # self-consistency check, not a fresh integral validation).
        Nvals = GaussianBasis.num_basis.(bs.shells)
        ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:bs.nshells]
        for iA = 1:length(atoms)
            dense = ∇ERI_2e4c(bs, iA)
            assembled = zeros(bs.nbas, bs.nbas, bs.nbas, bs.nbas, 3)
            for i in 1:bs.nshells, j in 1:bs.nshells, k in 1:bs.nshells, l in 1:bs.nshells
                Ni, Nj, Nk, Nl = Nvals[i], Nvals[j], Nvals[k], Nvals[l]
                I = (ao_offset[i]+1):(ao_offset[i]+Ni)
                J = (ao_offset[j]+1):(ao_offset[j]+Nj)
                K = (ao_offset[k]+1):(ao_offset[k]+Nk)
                L = (ao_offset[l]+1):(ao_offset[l]+Nl)
                assembled[I,J,K,L,:] .= ∇ERI_2e4c(bs, iA, i, j, k, l)
            end
            @test dense ≈ assembled atol=1e-10
        end
    end
end

aux = BasisSet("def2-universal-jkfit", atoms)

@testset "∂[ij|k]/∂X" begin
    for iA = 1:5
        dERI = ∇ERI_2e3c(bs, aux, iA)
        for k = 1:3
            @test dERI[:,:,:,k] ≈ ∇FD_ERI_2e3c(bs, aux, iA, k)
        end
    end
end

@testset "∂[i|j]/∂X" begin
    for iA = 1:5
        dERI = ∇ERI_2e2c(aux, iA)
        for k = 1:3
            @test dERI[:,:,k] ≈ ∇FD_ERI_2e2c(aux, iA, k)
        end
    end
end
