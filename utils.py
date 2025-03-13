import ufl
from ufl import sqrt
import numpy as np
import matplotlib.pyplot as plt
import ufl.constant

from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv
pv.start_xvfb()
# import sympy as sp
from pathlib import Path

import os
import time

def plotter_func(u=None, dim=2, mesh=None, title="", colorbar=False):
    V = u.ufl_function_space()
    if mesh is None:
        mesh = V.mesh
    topology, cell_types, geometry = plot.vtk_mesh(mesh)
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # VTK topology stored as (num_dofs_cell_0, dof_0,.... dof_(num_dofs_cell_0, num_dofs_cell_1, ....))
    # We assume we only have one cell type and one dofmap, thus every `num_dofs_per_cell` is the same
    num_dofs_per_cell = topology[0]
    # Get only dof indices
    topology_dofs = (np.arange(len(topology)) % (num_dofs_per_cell+1)) != 0

    # Map to global dof indices
    global_dofs = V.dofmap.index_map.local_to_global(topology[topology_dofs].copy())
    # Overwrite topology
    topology[topology_dofs] = global_dofs
    # Choose root
    root = 0
    # Gather data
    global_topology = mesh.comm.gather(topology[:(num_dofs_per_cell+1)*num_cells_local], root=root)
    global_geometry = mesh.comm.gather(geometry[:V.dofmap.index_map.size_local,:], root=root)
    global_ct = mesh.comm.gather(cell_types[:num_cells_local])
    global_vals = mesh.comm.gather(u.x.array[:num_dofs_local])
    if mesh.comm.rank == root:
        # Stack data
        root_geom = np.vstack(global_geometry)
        root_top = np.concatenate(global_topology)
        root_ct = np.concatenate(global_ct)
        root_vals = np.concatenate(global_vals)
        pv.OFF_SCREEN = True
        grid = pv.UnstructuredGrid(root_top, root_ct, root_geom)
        grid.point_data["u"] = root_vals
        grid.set_active_scalars("u")
        u_plotter = pv.Plotter(off_screen=True)
        u_plotter.add_mesh(grid)
        u_plotter.remove_scalar_bar()
        u_plotter.view_xy()
        u_plotter.camera.tight()
        if colorbar:
            u_plotter.add_scalar_bar()
        u_plotter.screenshot(f"{title}.png")

def plot_force_disp(B, name, out_file):
    plt.figure()
    B_ = np.array(B)
    plt.plot(B_[:, 1], np.abs(B_[:, 0]*1e-3))
    plt.savefig(f"{out_file}/force_disp_{name}.png")
    np.savetxt(f"{out_file}/force_{name}.txt", B_)
    plt.close()

def distance_points_to_segment(points, x1, y1, x2, y2):
    # Convert the points array to a numpy array for vectorized operations
    points = np.array(points)
    
    # Define the vectors
    AB = np.array([x2 - x1, y2 - y1])
    AB_AB = np.dot(AB, AB)
    
    distances = []
    for point in points:
        px, py = point
        
        # Vector from A to P
        AP = np.array([px - x1, py - y1])
        
        # Project vector AP onto vector AB
        AP_AB = np.dot(AP, AB)
        t = AP_AB / AB_AB
        
        # Find the closest point on the segment
        if t < 0:
            closest_point = np.array([x1, y1])
        elif t > 1:
            closest_point = np.array([x2, y2])
        else:
            closest_point = np.array([x1, y1]) + t * AB
        
        # Compute the distance from the point to the closest point
        distance = np.linalg.norm(np.array([px, py]) - closest_point)
        distances.append(distance)
    
    return np.array(distances)