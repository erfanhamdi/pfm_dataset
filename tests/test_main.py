import os
import dolfinx
import ufl
import mpi4py

def test_save_data():
    os.system("python src/main.py --prefix test_save_data --num_steps 1")
    assert os.path.exists("results/test_save_data/1726/p_unit.xdmf")
    assert os.path.exists("results/test_save_data/1726/u_unit.xdmf")
    assert os.path.exists("results/test_save_data/1726/p_unit.h5")
    assert os.path.exists("results/test_save_data/1726/u_unit.h5")
    os.system("rm -rf results/test_save_data")
