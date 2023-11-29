"""
Test energy computations
"""
import quax
import psi4
import pytest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os

molecule = psi4.geometry("""
0 1
H   -1.424097055410    -0.993053750648     0.000000000000
H    1.424209276385    -0.993112599269     0.000000000000
units bohr
""")
basis_name = 'sto-3g'

#quax_e = quax.core.energy(molecule, basis_name, "hf")
geom2d = np.asarray(molecule.geometry())
geom_list = geom2d.reshape(-1).tolist()
geom = jnp.asarray(geom2d.flatten())
dim = geom.reshape(-1).shape[0]
mult = molecule.multiplicity()
xyz_file_name = "geom.xyz"
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
charge = molecule.molecular_charge()
nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
options = {'integral_algo': "quax_core", 'maxit': 100,'damping': False,
                       'damp_factor': 0.5,
                       'spectral_shift': True,}
args = (geom, basis_name, xyz_path, nuclear_charges, charge, options)
print (args)
print(quax.methods.hartree_fock.restricted_hartree_fock(*args))
