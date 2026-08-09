atoms = Molecules.parse_string("""
C   -2.131551124300    2.286168823700    0.000000000000
H   -1.061551124300    2.286168823700    0.000000000000
H   -2.488213906200    1.408104616400    0.496683911300
H   -2.488218762100    2.295059432700   -1.008766153900
H   -2.488220057000    3.155340844300    0.512081313000""")

bs = BasisSet("cc-pvdz", atoms)

@testset "∂²S/∂X²" begin
    for iA = 1:length(atoms), iB = 1:length(atoms)
        d2S = ∇2overlap(bs, iA, iB)
        for k = 1:3
            @test d2S[:,:,:,k] ≈ ∇2FD_overlap(bs, iA, iB, k) atol=1e-6
        end
    end
end

@testset "∂²T/∂X²" begin
    for iA = 1:length(atoms), iB = 1:length(atoms)
        d2T = ∇2kinetic(bs, iA, iB)
        for k = 1:3
            @test d2T[:,:,:,k] ≈ ∇2FD_kinetic(bs, iA, iB, k) atol=1e-6
        end
    end
end

@testset "∂²V/∂X²" begin
    for iA = 1:length(atoms), iB = 1:length(atoms)
        d2V = ∇2nuclear(bs, iA, iB)
        for k = 1:3
            @test d2V[:,:,:,k] ≈ ∇2FD_nuclear(bs, iA, iB, k) atol=1e-6
        end
    end
end

@testset "∂²V/∂X² translational invariance" begin
    # sum_B H(A,B) == 0 exactly (to machine precision), for any fixed A --
    # the whole molecule (shells + all nuclear charges) is translation
    # invariant, at every derivative order. Independent of finite
    # differences, so it catches a different class of bug (global sign/axis
    # errors that a per-entry FD comparison can also miss if consistently
    # applied on both sides).
    for iA = 1:length(atoms)
        total = zeros(bs.nbas, bs.nbas, 3, 3)
        for iB = 1:length(atoms)
            total .+= ∇2nuclear(bs, iA, iB)
        end
        @test maximum(abs.(total)) < 1e-8
    end
end

auxbs = BasisSet("cc-pvqz-jkfit", atoms)

@testset "∂²(P|Q)/∂X²" begin
    for iA = 1:length(atoms), iB = 1:length(atoms)
        d2J = ∇2ERI_2e2c(auxbs, iA, iB)
        for k = 1:3
            @test d2J[:,:,:,k] ≈ ∇2FD_ERI_2e2c(auxbs, iA, iB, k) atol=1e-6
        end
    end
end

@testset "∂²(μν|P)/∂X²" begin
    for iA = 1:length(atoms), iB = 1:length(atoms)
        d2P = ∇2ERI_2e3c(bs, auxbs, iA, iB)
        for k = 1:3
            @test d2P[:,:,:,:,k] ≈ ∇2FD_ERI_2e3c(bs, auxbs, iA, iB, k) atol=1e-6
        end
    end
end

@testset "2-electron Hessian translational invariance" begin
    # Same sum_B H(A,B) == 0 check as ∂²V/∂X² above, for both the metric and
    # 3-center Hessians.
    for iA = 1:length(atoms)
        totalJ = zeros(auxbs.nbas, auxbs.nbas, 3, 3)
        totalP = zeros(bs.nbas, bs.nbas, auxbs.nbas, 3, 3)
        for iB = 1:length(atoms)
            totalJ .+= ∇2ERI_2e2c(auxbs, iA, iB)
            totalP .+= ∇2ERI_2e3c(bs, auxbs, iA, iB)
        end
        @test maximum(abs.(totalJ)) < 1e-8
        @test maximum(abs.(totalP)) < 1e-8
    end
end

@testset "(μν|P) mu<->nu AO symmetry" begin
    # (μν|P) = (νμ|P) exactly, at every derivative order.
    for iA = 1:length(atoms), iB = 1:length(atoms)
        H = ∇2ERI_2e3c(bs, auxbs, iA, iB)
        @test H ≈ permutedims(H, (2,1,3,4,5))
    end
end

@testset "∂²(μν|λσ)/∂X² shell-quartet" begin
    # Shell-quartet-level dense 4-center ERI Hessian (no whole-array
    # counterpart -- ∇2ERI_2e4c is meant to be called per surviving quartet
    # by a caller doing its own screening, see this function's docstring).
    # Bounded shell range to keep this test fast -- exhaustive nshells^4 x
    # natm^2 isn't needed to catch an orientation/indexing bug.
    nsh = min(bs.nshells, 4)
    for iA = 1:length(atoms), iB = 1:length(atoms)
        for i = 1:nsh, j = 1:nsh, k = 1:nsh, l = 1:nsh
            d2 = ∇2ERI_2e4c(bs, iA, iB, i, j, k, l)
            for dir = 1:3
                @test d2[:,:,:,:,:,dir] ≈ ∇2FD_ERI_2e4c(bs, iA, iB, i, j, k, l, dir) atol=1e-6
            end
        end
    end
end
