import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import host_callback
import numpy as np
import psi4

from .ints import compute_integrals
from .energy_utils import nuclear_repulsion, cholesky_orthogonalization


def restricted_hartree_fock(
    geom,
    basis_name,
    xyz_path,
    nuclear_charges,
    charge,
    options,
    deriv_order=0,
    return_aux_data=False,
    dmguess=None,
):
    # Load keyword options
    maxit = options["maxit"]
    damping = options["damping"]
    damp_factor = options["damp_factor"]
    spectral_shift = options["spectral_shift"]
    convergence = 1e-10

    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    # If we are doing MP2 or CCSD after, might as well use jit-compiled JK-build, since HF will not be memory bottleneck
    if return_aux_data:
        jk_build = jax.jit(
            jax.vmap(
                jax.vmap(
                    lambda x, y: jnp.tensordot(x, y, axes=[(0, 1), (0, 1)]),
                    in_axes=(0, None),
                ),
                in_axes=(0, None),
            )
        )
    else:
        jk_build = jax.vmap(
            jax.vmap(
                lambda x, y: jnp.tensordot(x, y, axes=[(0, 1), (0, 1)]),
                in_axes=(0, None),
            ),
            in_axes=(0, None),
        )

    host_callback.id_print(maxit, what="do integrals")
    S, T, V, G = compute_integrals(
        geom, basis_name, xyz_path, nuclear_charges, charge, deriv_order, options
    )
    # Canonical orthogonalization via cholesky decomposition
    host_callback.id_print(maxit, what="cholesky")
    A = cholesky_orthogonalization(S)

    nbf = S.shape[0]

    # For slightly shifting eigenspectrum of transformed Fock for degenerate eigenvalues
    # (JAX cannot differentiate degenerate eigenvalue eigh)
    if spectral_shift:
        # Shifting eigenspectrum requires lower convergence.
        convergence = 1e-8
        fudge = jnp.asarray(np.linspace(0, 1, nbf)) * convergence
        shift = jnp.diag(fudge)
    else:
        shift = jnp.zeros_like(S)

    host_callback.id_print(maxit, what="hamiltonian")
    H = T + V
    Enuc = nuclear_repulsion(geom.reshape(-1, 3), nuclear_charges)
    #host_callback.id_print(dmguess, what="dmguess")
    if dmguess is None:
        D = jnp.zeros_like(H)
    else:
        D = jnp.array(dmguess)
    

    def rhf_iter(F, D):
        E_scf = jnp.einsum("pq,pq->", F + H, D) #+ Enuc
        Fp = jnp.dot(A.T, jnp.dot(F, A))
        Fp = Fp + shift
        eps, C2 = jnp.linalg.eigh(Fp)
        C = jnp.dot(A, C2)
        Cocc = C[:, :ndocc]
        D = jnp.dot(Cocc, Cocc.T)
        return E_scf, D, C, eps

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    Dold = D
    dRMS = 1.0

    # Converge according to energy and DIIS residual to ensure eigenvalues and eigenvectors are maximally converged.
    # This is crucial for numerical stability for higher order derivatives of correlated methods.
    while (abs(E_scf - E_old) > convergence) or (dRMS > convergence):
        host_callback.id_print(D, what="dm current")
        E_old = E_scf * 1
        if damping:
            if iteration < 10:
                D = Dold * damp_factor + D * damp_factor
                Dold = D * 1
        # Build JK matrix: 2 * J - K
        JK = 2 * jk_build(G, D)
        JK -= jk_build(G.transpose((0, 2, 1, 3)), D)
        # Build Fock
        F = H + JK
        # Update convergence error
        if iteration > 1:
            diis_e = jnp.einsum("ij,jk,kl->il", F, D, S) - jnp.einsum(
                "ij,jk,kl->il", S, D, F
            )
            diis_e = A.dot(diis_e).dot(A)
            dRMS = jnp.mean(diis_e ** 2) ** 0.5
        # Compute energy, transform Fock and diagonalize, get new density
        host_callback.id_print(iteration, what="iter")
        E_scf, D, C, eps = rhf_iter(F, D)
        iteration += 1
        if iteration == maxit:
            break
    print(iteration, " RHF iterations performed")

    # If many orbitals are degenerate, warn that higher order derivatives may be unstable
    tmp = jnp.round(eps, 6)
    ndegen_orbs = tmp.shape[0] - jnp.unique(tmp).shape[0]
    if (ndegen_orbs / nbf) > 0.20:
        print(
            "Hartree-Fock warning: More than 20% of orbitals have degeneracies. Higher order derivatives may be unstable due to eigendecomposition AD rule"
        )
    if not return_aux_data:
        return E_scf
    else:
        return E_scf, C, eps, G
