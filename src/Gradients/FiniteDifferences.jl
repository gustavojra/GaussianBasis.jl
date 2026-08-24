# Given a list of atoms, create a cartesian displacement in one of them (A).
# Rebuilds `atoms`/`shells` in memory instead of going through
# `BasisSet(name, atoms)` -- shell composition, exponents, and contraction
# coefficients never change when only a nuclear position moves, so there is
# no need to re-read the basis-set library file from disk (which
# `BasisSet(name, atoms)` does, once per atom, every single call -- the
# exact pattern that was exhausting the file-descriptor limit in CI, see
# `extract_atom_from_bs`). The backend-specific `lib` is patched to match
# rather than recomputed from scratch (see `displaced_lib` below).
function create_displacement(BS::BasisSet, A::Int, i::Int, h)

    disp = zeros(3)
    disp[i] += h

    bs_plus = displaced_basis_set(BS, A, disp)
    bs_minus = displaced_basis_set(BS, A, -disp)

    return bs_plus, bs_minus
end

function displaced_basis_set(BS::BasisSet, A::Int, disp::AbstractVector)
    old_atom = BS.atoms[A]
    new_atom = Atom(old_atom.Z, old_atom.mass, old_atom.xyz + disp)

    new_atoms = copy(BS.atoms)
    new_atoms[A] = new_atom

    # Every shell keeps its own literal copy of its atom (Atom is isbits, so
    # this is a value, not a shared reference -- see on_atom_flags). Only the
    # shells actually on atom A need a new atom field; everything else about
    # them (l, coef, exp) is untouched.
    new_shells = map(BS.shells) do shell
        shell.atom == old_atom ? typeof(shell)(shell.l, shell.coef, shell.exp, new_atom) : shell
    end

    new_lib = displaced_lib(BS.lib, A, new_atom)

    return BasisSet(BS.name, new_atoms, new_shells, BS.basis_per_atom, BS.shells_per_atom,
                     BS.natoms, BS.nbas, BS.nshells, new_lib)
end

# ACSint computes directly from `atoms`/`shells` at call time -- no
# precomputed geometry cache to patch.
displaced_lib(lib::ACSint, ::Int, ::Atom) = lib

# LCint caches every atom's position in 3 slots of its flat `env` array
# (`lib.atm[2 + ATM_SLOTS*(A-1)]` is the 0-based offset libcint itself
# recorded for atom A when `env` was built -- see `LCint`'s constructor in
# Libs.jl). `atm`/`bas` (atomic numbers, shell-to-atom mapping, angular
# momenta, and the exponent/coefficient offsets) are position-independent
# and can be reused unchanged; only those 3 `env` entries need updating.
function displaced_lib(lib::LCint, A::Int, new_atom::Atom)
    ATM_SLOTS = 6
    new_env = copy(lib.env)
    off = lib.atm[2 + ATM_SLOTS*(A-1)]
    new_env[off+1:off+3] .= new_atom.xyz ./ Molecules.bohr_to_angstrom
    return LCint(lib.atm, lib.natm, lib.bas, lib.nbas, new_env)
end

function ∇FD_overlap(BS::BasisSet, A, i, h = 1e-5)
    return ∇FD_1e(BS, overlap, A, i, h)
end

function ∇FD_kinetic(BS::BasisSet, A, i, h = 1e-5)
    return ∇FD_1e(BS, kinetic, A, i, h)
end

function ∇FD_nuclear(BS::BasisSet, A, i, h = 1e-5)
    return ∇FD_1e(BS, nuclear, A, i, h)
end

function ∇FD_1e(BS::BasisSet, callback, A, i, h)

    bs_plus, bs_minus = create_displacement(BS, A, i, h)

    Xplus = callback(bs_plus)
    Xminus = callback(bs_minus)

    return (Xplus - Xminus) ./ (2*h/Molecules.bohr_to_angstrom)
end

function ∇FD_ERI_2e4c(BS::BasisSet, A, i, h=1e-5)

    bs_plus, bs_minus = create_displacement(BS, A, i, h)

    Xplus = ERI_2e4c(bs_plus)
    Xminus = ERI_2e4c(bs_minus)

    return (Xplus - Xminus) ./ (2*h/Molecules.bohr_to_angstrom)

end

function ∇FD_sparseERI_2e4c(BS::BasisSet, A, i, h=1e-5)

    bs_plus, bs_minus = create_displacement(BS, A, i, h)

    Iplus, Xplus = sparseERI_2e4c(bs_plus)
    Iminus, Xminus = sparseERI_2e4c(bs_minus)

    @assert Iplus == Iminus

    Xout = (Xplus - Xminus) ./ (2*h/Molecules.bohr_to_angstrom)

    return Iplus, Xout
end

function ∇FD_ERI_2e3c(BS::BasisSet, auxBS::BasisSet, A, i, h=1e-5)

    bs_plus, bs_minus = create_displacement(BS, A, i, h)
    Abs_plus, Abs_minus = create_displacement(auxBS, A, i, h)

    Xplus  = ERI_2e3c(bs_plus, Abs_plus)
    Xminus = ERI_2e3c(bs_minus, Abs_minus)

    return (Xplus - Xminus) ./ (2*h/Molecules.bohr_to_angstrom)

end

function ∇FD_ERI_2e2c(BS::BasisSet, A, i, h=1e-5)

    bs_plus, bs_minus = create_displacement(BS, A, i, h)

    Xplus = ERI_2e2c(bs_plus)
    Xminus = ERI_2e2c(bs_minus)

    return (Xplus - Xminus) ./ (2*h/Molecules.bohr_to_angstrom)

end