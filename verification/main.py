import ufl
import numpy as np
import ufl.constant

from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path

from src.utils import plotter_func, plot_force_disp, distance_points_to_segment
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='2D shear benchmark test')
parser.add_argument('--case', type=str, default="tension", help='Case to run tension or shear')
parser.add_argument('--model', type=str, default="miehe", help='Model to use')
parser.add_argument('--mesh_size', type=int, default=100, help='Mesh size')
parser.add_argument('--out_file', type=str, default="tension-verification", help='Output file')
parser.add_argument('--job_id', type=int, default=0, help='Job id')
parser.add_argument('--l_c', type = float , default=0.15, help='l_c value')
args = parser.parse_args()

sim_case = args.case
model = args.model
mesh_size = args.mesh_size
out_file_arg = args.out_file
job_id = args.job_id
l_c = args.l_c

ksp = PETSc.KSP.Type.GMRES
pc = PETSc.PC.Type.HYPRE

out_file = f"./results/{out_file_arg}"
Path(out_file).mkdir(parents=True, exist_ok=True)

print(f"2D {sim_case} benchmark test, out_file = {out_file}")

results_folder = Path(out_file)
results_folder.mkdir(exist_ok=True, parents=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t_transition = 500
if sim_case == "tension":
    # for a coarser mesh you can uncomment this line
    mesh_address = "./tension_mesh_no_notch_001.msh"
    # domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [mesh_size, mesh_size], cell_type=mesh.CellType.quadrilateral)
    domain, _, _ = gmshio.read_from_msh(mesh_address, MPI.COMM_WORLD, gdim=2)
    delta_T1 = fem.Constant(domain, 1e-5)
    delta_T2 = fem.Constant(domain, 1e-6)
    num_steps = 2000
elif sim_case == "shear":
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [mesh_size, mesh_size], cell_type=mesh.CellType.quadrilateral)
    delta_T1 = fem.Constant(domain, 1e-5)
    delta_T2 = fem.Constant(domain, 1e-5)
    num_steps = 2000
elif sim_case == "internal":
    mesh_address = "/projectnb/lejlab2/erfan/pfm_ds/verification/internal_regional.msh"
    domain, _, _ = gmshio.read_from_msh(mesh_address, MPI.COMM_WORLD, gdim=2)
    delta_T1 = fem.Constant(domain, 5e-7)
    delta_T2 = fem.Constant(domain, 5e-7)
    num_steps = 6000

if sim_case == "tension" or sim_case == "shear":
    G_c_ = fem.Constant(domain, 2.7)
    l_0_ = fem.Constant(domain, 4e-3)
    E = fem.Constant(domain, 210.0e3)
    nu = fem.Constant(domain, 0.3)
    top_bound = 0.5
    left_bound = -0.5
    right_bound = 0.5
    bottom_bound = -0.5
if sim_case == "internal":
    G_c_ = fem.Constant(domain, 3e-3)
    l_0_ = fem.Constant(domain, l_c)
    E = fem.Constant(domain, 30.0e3)
    nu = fem.Constant(domain, 0.333)
    top_bound = 50.0
    left_bound = 0.0
    right_bound = 50.0
    bottom_bound = 0.0

mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
n = fem.Constant(domain, 3.0)
Kn = lmbda + 2 * mu / n
gamma_star = fem.Constant(domain, 5.0)

t_ = fem.Constant(domain, 0.0)

out_file_name = XDMFFile(domain.comm, f"{out_file}/p_unit22.xdmf", 'w')
out_file_name.write_mesh(domain)
out_file_name_u = XDMFFile(domain.comm, f"{out_file}/u_unit22.xdmf", 'w')
out_file_name_u.write_mesh(domain)

# Defining function spaces for displacement, phase and history field.
V = fem.functionspace(domain, ("Lagrange", 1,))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
VV = fem.functionspace(domain, ("DG", 0,))

u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
p, q = ufl.TrialFunction(V), ufl.TestFunction(V)

u_new, u_old = fem.Function(W), fem.Function(W)
p_new, H_old, p_old = fem.Function(V), fem.Function(VV), fem.Function(V)
H_init_ = fem.Function(V)

tdim = domain.topology.dim
fdim = tdim - 1

############################################ defining the boundary conditions ########################################
def top_boundary(x):
    return np.isclose(x[1], top_bound)

def left_boundary(x):
    return np.isclose(x[0], left_bound)

def right_boundary(x):
    return np.isclose(x[0], right_bound)

def bottom_boundary(x):
    return np.isclose(x[1], bottom_bound)

top_facet = mesh.locate_entities_boundary(domain, fdim, top_boundary)
top_marker = 1
top_marked_facets = np.full_like(top_facet, top_marker)

bot_facet = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
bot_marker = 2
bot_marked_facets = np.full_like(bot_facet, bot_marker)

right_facet = mesh.locate_entities_boundary(domain, fdim, right_boundary)
right_marker = 3
right_marked_facets = np.full_like(right_facet, right_marker)

left_facet = mesh.locate_entities_boundary(domain, fdim, left_boundary)
left_marker = 4
left_marked_facets = np.full_like(left_facet, left_marker)

marked_facets = np.hstack([top_facet, bot_facet, right_facet, left_facet])
marked_values = np.hstack([np.full_like(top_facet, 1), np.full_like(bot_facet, 2), np.full_like(right_facet, 3), np.full_like(left_facet, 4)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

top_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, top_facet)
top_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, top_facet)

bot_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, bot_facet)
bot_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, bot_facet)

right_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, right_facet)
right_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, right_facet)

left_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, left_facet)
left_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, left_facet)

u_bc_top = fem.Constant(domain, default_scalar_type(0.0))
u_bc_bot_ = fem.Constant(domain, default_scalar_type(0.0))
u_bc_right = fem.Constant(domain, default_scalar_type(0.0))

bc_bot_y = fem.dirichletbc(default_scalar_type(0.0), bot_y_dofs, W.sub(1))
bc_bot_x = fem.dirichletbc(default_scalar_type(0.0), bot_x_dofs, W.sub(0))
bc_top_y = fem.dirichletbc(u_bc_top, top_y_dofs, W.sub(1))
bc_left_x = fem.dirichletbc(default_scalar_type(0.0), left_x_dofs, W.sub(0))
bc_left_y = fem.dirichletbc(default_scalar_type(0.0), left_y_dofs, W.sub(1))
bc_right_x = fem.dirichletbc(default_scalar_type(0.0), right_x_dofs, W.sub(0))

if sim_case == "tension":
    bc = [bc_bot_y, bc_bot_x, bc_top_y]
elif sim_case == "shear":
    bc_top_x = fem.dirichletbc(u_bc_top, top_x_dofs, W.sub(0))
    bc_top_y = fem.dirichletbc(default_scalar_type(0.0), top_y_dofs, W.sub(1))
    bc = [bc_bot_y, bc_bot_x, bc_top_y, bc_top_x]
elif sim_case == "internal":
    bc_bot_y = fem.dirichletbc(u_bc_bot_, bot_y_dofs, W.sub(1))
    bc = [bc_bot_y, bc_top_y]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda*ufl.tr(epsilon(u))*ufl.Identity(2) + 2.0*mu*epsilon(u)

def bracket_pos(u):
    return 0.5*(u + np.abs(u))

def bracket_neg(u):
    return 0.5*(u - np.abs(u))

# Spectral decomposition of the strain tensor
A = ufl.variable(epsilon(u_new))
I1 = ufl.tr(A)
delta = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0] + 3.0e-16 ** 2
eigval_1 = (I1 - ufl.sqrt(delta)) / 2
eigval_2 = (I1 + ufl.sqrt(delta)) / 2
eigvec_1 = ufl.diff(eigval_1, A).T
eigvec_2 = ufl.diff(eigval_2, A).T
epsilon_p = 0.5 * (eigval_1 + abs(eigval_1)) * eigvec_1 + 0.5 * (eigval_2 + abs(eigval_2)) * eigvec_2
epsilon_n = 0.5 * (eigval_1 - abs(eigval_1)) * eigvec_1 + 0.5 * (eigval_2 - abs(eigval_2)) * eigvec_2

def strain_dev(u):
    return epsilon(u) - (1/3) * ufl.tr(epsilon(u)) * ufl.Identity(2)

# Different energy decompositions
def psi_pos_m(u):
    return 0.5*lmbda*(bracket_pos(ufl.tr(epsilon(u)))**2) + mu*(ufl.inner(epsilon_p, epsilon_p))

def psi_neg_m(u):
    return 0.5*lmbda*(bracket_neg(ufl.tr(epsilon(u)))**2) + mu*(ufl.inner(epsilon_n, epsilon_n))

def psi_pos_a(u):
    return 0.5 * Kn * bracket_pos(ufl.tr(epsilon(u)))**2 + mu * ufl.inner(strain_dev(u), strain_dev(u))

def psi_neg_a(u):
    return 0.5 * Kn * bracket_neg(ufl.tr(epsilon(u)))**2

def psi_pos_s(u):
    return mu * ufl.inner(strain_dev(u), strain_dev(u)) + 0.5 * Kn * (bracket_pos(ufl.tr(epsilon(u)))**2 - gamma_star * bracket_neg(ufl.tr(epsilon(u)))**2)

def psi_neg_s(u):
    return (1 + gamma_star) * 0.5 * Kn * bracket_neg(ufl.tr(epsilon(u)))**2

if model == "miehe":
    psi_pos = psi_pos_m(u_new)
    psi_neg = psi_neg_m(u_new)
elif model == "amor":
    psi_pos = psi_pos_a(u_new)
    psi_neg = psi_neg_a(u_new)
elif model == "star":
    psi_pos = psi_pos_s(u_new)
    psi_neg = psi_neg_s(u_new)

def H(u_new, H_old):
    return ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old)
############################################# defining the initial cracks ########################################
def H_init(dist_list, l_0, G_c):
    distances = np.array(dist_list)
    distances = np.min(distances, axis=0)
    mask0 = distances <= l_0.value/2
    H = np.zeros_like(distances)
    phi_c = 0.999
    H[mask0] = ((phi_c/(1-phi_c))*G_c.value/(2*l_0.value))*(1-(2*distances[mask0]/l_0.value))
    return H

if sim_case == "internal":
    A_ = [[12.5, 17.5], [22.5, 22.5], [32.5, 27.5]]
    B_ = [[17.5, 22.5], [27.5, 27.5], [37.5, 32.5]]
    points = domain.geometry.x[:, :2]
    dist_list = []

    for idx in range(len(A_)):
        distances = distance_points_to_segment(points, A_[idx][0], A_[idx][1], B_[idx][0], B_[idx][1])
        dist_list.append(distances)

    H_init_.x.array[:] = H_init(dist_list, l_0_, G_c_)
    H_old.interpolate(H_init_)

elif sim_case == "tension" or sim_case == "shear":
    A_ = [[-0.5, 0.0]]
    B_ = [[0.0, 0.0]]
    points = domain.geometry.x[:, :2]
    dist_list = []

    for idx in range(len(A_)):
        distances = distance_points_to_segment(points, A_[idx][0], A_[idx][1], B_[idx][0], B_[idx][1])
        dist_list.append(distances)

    H_init_.x.array[:] = H_init(dist_list, l_0_, G_c_)
    H_old.interpolate(H_init_)

#################################### problem definition ############################################
T = fem.Constant(domain, default_scalar_type((0, 0)))
E_du = ((1.0-p_new)**2)*ufl.inner(ufl.grad(v),sigma(u))*dx + ufl.dot(T, v) * ds
a_u = fem.form(ufl.lhs(E_du))
L_u = fem.form(ufl.rhs(E_du))
A_u = create_matrix(a_u)
b_u = create_vector(L_u)

solver_u = PETSc.KSP().create(domain.comm)
solver_u.setOperators(A_u)
solver_u.setType(ksp)
solver_u.getPC().setType(pc)

E_phi = (((l_0_**2) * ufl.dot(ufl.grad(p), ufl.grad(q))) + ((2*l_0_/G_c_) * H(u_new, H_old) +1 ) * p * q )* dx - (2*l_0_/G_c_) * H(u_new, H_old) * q * dx
a_phi = fem.form(ufl.lhs(E_phi))
L_phi = fem.form(ufl.rhs(E_phi))
A_phi = create_matrix(a_phi)
b_phi = create_vector(L_phi)

solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A_phi)
solver_phi.setType(ksp)
solver_phi.getPC().setType(pc)

u_l2_error = fem.form(ufl.dot(u_new - u_old, u_new - u_old)*dx)
p_l2_error = fem.form(ufl.dot(p_new - p_old, p_new - p_old)*dx)

############################ new rxn force calculation ###############################################
B_bot = []
residual = ufl.action(ufl.lhs(E_du), u_new) - ufl.rhs(E_du)
v_reac = fem.Function(W)
virtual_work_form = fem.form(ufl.action(residual, v_reac))

bot_dofs = fem.locate_dofs_geometrical(W, bottom_boundary)
u_bc_bot = fem.Function(W)
bc_bot_rxn = fem.dirichletbc(u_bc_bot, bot_dofs)
bc_rxn = [bc_bot_rxn]

def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

if sim_case == "tension" or sim_case == "internal":
    u_bc_bot.sub(1).interpolate(one)
else:
    u_bc_bot.sub(0).interpolate(one)

################################# main simulation loop ###############################################
delta_T = delta_T1
error_tol = fem.Constant(domain, 1e-4)
error_total = fem.Constant(domain, 1.0)
H_expr = fem.Expression(ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old), VV.element.interpolation_points())
t_keeper = 0
for i in range(num_steps+1):
    if rank == 0:
        print(f"Step = {i}/{num_steps}", flush=True)
    if i > t_transition:
        delta_T = delta_T2
    t_.value += delta_T.value
    t_keeper += delta_T.value
    u_bc_top.value = t_.value
    if case=="internal":
        u_bc_bot_.value = -t_.value
    error_total.value = 1.0
    flag = 1
    staggered_iter = 0
    while flag:
        staggered_iter +=1
        if error_total.value < error_tol.value:
            flag = 0
            break
        A_u.zeroEntries()
        assemble_matrix(A_u, a_u, bcs = bc)
        A_u.assemble()
        with b_u.localForm() as loc:
            loc.set(0)
        assemble_vector(b_u, L_u)
        apply_lifting(b_u, [a_u], [bc])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_u, bc)
        solver_u.solve(b_u, u_new.vector)
        u_new.x.scatter_forward()
        
        A_phi.zeroEntries()
        assemble_matrix(A_phi, a_phi, bcs = [])
        A_phi.assemble()
        with b_phi.localForm() as loc:
            loc.set(0)
        assemble_vector(b_phi, L_phi)
        b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver_phi.solve(b_phi, p_new.vector)
        p_new.x.scatter_forward()
        
        error_total.value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(u_l2_error), op=MPI.SUM)) + np.sqrt(domain.comm.allreduce(fem.assemble_scalar(p_l2_error), op=MPI.SUM))
        if rank == 0:
            print(f"staggered_iter = {staggered_iter}, error total = {error_total.value}", flush=True)
        p_old.x.array[:] = p_new.x.array
        u_old.x.array[:] = u_new.x.array
        H_old.interpolate(H_expr)
    ################################################################################
    v_reac.vector.set(0.0)
    v_reac.x.scatter_forward()
    fem.set_bc(v_reac.vector, [bc_bot_rxn])
    R_bot_y= domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

    if domain.comm.rank == 0:
        B_bot.append([np.sum(R_bot_y), t_keeper])
    if i%100 == 0:
        out_file_name.write_function(p_new, t_keeper)
        out_file_name_u.write_function(u_new, t_keeper)
        plotter_func(p_new, dim=1, mesh = domain, title=f"{out_file}/p_{i}")
        if rank == 0:
            print(f"Step = {i},  iter = {staggered_iter}, error total = {error_total.value}", flush=True)
            plot_force_disp(B_bot, "bot_rxn", out_file)
################################################################################
out_file_name.close()
out_file_name_u.close()
if rank == 0:
    print(f"rank = {rank} done.")
