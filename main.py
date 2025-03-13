import ufl
import numpy as np
import ufl.constant
import glob
import time
import os

from dolfinx import mesh, fem, io, default_scalar_type, log
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path

from utils import plotter_func, plot_force_disp, distance_points_to_segment
import argparse

# Configure HDF5 settings for MPI environment
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Disable file locking
os.environ["HDF5_MPI_OPS_COLLECTIVE"] = "TRUE" # Enable collective operations

parser = argparse.ArgumentParser(description='2D internal cracks')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--model', type=str, default="miehe", help='Model to use')
parser.add_argument('--mesh_size', type=int, default=100, help='Mesh size')
parser.add_argument('--prefix', type=str, default="test", help='Output file')
parser.add_argument('--sim_case', type=str, default="shear", help='Simulation case')
parser.add_argument('--job_id', type=int, default=0, help='Job id')
parser.add_argument('--g_c', type=float, default=1.0, help='G_c')
parser.add_argument('--e_', type=float, default=1000.0e3, help='E')
parser.add_argument('--domain_size', type=float, default=2.0, help='Domain size')
args = parser.parse_args()

seed = args.seed - 1
model = args.model
mesh_size = args.mesh_size
prefix = args.prefix
sim_case = args.sim_case
job_id = args.job_id
g_c_ = args.g_c
e_ = args.e_
domain_size = args.domain_size

start_time = time.time()

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Running on {size} MPI processes")
    print(f"Job id: {job_id}")

# Setting KSP solver and preconditioner
ksp = PETSc.KSP.Type.GMRES
pc = PETSc.PC.Type.HYPRE
rtol = 1e-8
max_it = 1000

initial_cracks = glob.glob("./initial_cracks/*.npy")
initial_cracks = sorted(initial_cracks, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
crack_pattern = np.load(initial_cracks[seed])
seed_val = int(initial_cracks[seed].split("/")[-1].split(".")[0])
out_file = f"./results/{prefix}/{seed_val}"
results_folder = Path(out_file)

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([domain_size, domain_size])], [mesh_size, mesh_size], cell_type=mesh.CellType.quadrilateral)

if rank == 0:
    print(f"seed: {seed_val}, out_file = {out_file}", flush=True)
    Path(out_file).mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(exist_ok=True, parents=True)


# Initialize output file objects
try:
    # Try to initialize XDMFFile objects for the base mesh
    out_file_name = io.XDMFFile(domain.comm, f"{out_file}/p_unit.xdmf", 'w')
    out_file_name.write_mesh(domain)
    out_file_name_u = io.XDMFFile(domain.comm, f"{out_file}/u_unit.xdmf", 'w')
    out_file_name_u.write_mesh(domain)
    # Set flag to indicate that file initialization succeeded
    file_init_success = True
except RuntimeError as e:
    if rank == 0:
        print(f"Warning: Could not initialize output files: {str(e)}")
        print("Will try individual file writes during simulation.")
    file_init_success = False

delta_T1 = fem.Constant(domain, 1e-6)
G_c_ = fem.Constant(domain, g_c_)
l_0_ = fem.Constant(domain, 0.01)
E = fem.Constant(domain, e_)
nu = fem.Constant(domain, 0.3)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
n = fem.Constant(domain, 3.0)
Kn = lmbda + 2 * mu / n
gamma_star = fem.Constant(domain, 5.0)

num_steps = 5000
save_freq = 50
if sim_case == "shear":
    num_steps = 10000
    save_freq = 100
t_ = fem.Constant(domain, 0.0)

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
    return np.isclose(x[1], domain_size)

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], domain_size)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

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
top_phi_dofs = fem.locate_dofs_topological(V, fdim, top_facet)

bot_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, bot_facet)
bot_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, bot_facet)
bot_phi_dofs = fem.locate_dofs_topological(V, fdim, bot_facet)
right_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, right_facet)
right_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, right_facet)
right_phi_dofs = fem.locate_dofs_topological(V, fdim, right_facet)

left_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, left_facet)
left_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, left_facet)
left_phi_dofs = fem.locate_dofs_topological(V, fdim, left_facet)
u_bc_top = fem.Constant(domain, default_scalar_type(0.0))
u_bc_right = fem.Constant(domain, default_scalar_type(0.0))

bc_bot_y = fem.dirichletbc(default_scalar_type(0.0), bot_y_dofs, W.sub(1))
bc_bot_x = fem.dirichletbc(default_scalar_type(0.0), bot_x_dofs, W.sub(0))

if sim_case == "tension":
    bc_top_y = fem.dirichletbc(u_bc_top, top_y_dofs, W.sub(1))
    bc_right_x = fem.dirichletbc(u_bc_right, right_x_dofs, W.sub(0))
    bc_left_x = fem.dirichletbc(default_scalar_type(0.0), left_x_dofs, W.sub(0))
    bc_left_y = fem.dirichletbc(default_scalar_type(0.0), left_y_dofs, W.sub(1))
    bc = [bc_bot_y, bc_top_y, bc_right_x, bc_left_x]
    bc_phi = []
else:
    bc_top_x = fem.dirichletbc(u_bc_top, top_x_dofs, W.sub(0))
    bc_top_y = fem.dirichletbc(default_scalar_type(0.0), top_y_dofs, W.sub(1))
    bc_phi_top = fem.dirichletbc(default_scalar_type(0.0), top_phi_dofs, V)
    bc_phi_bot = fem.dirichletbc(default_scalar_type(0.0), bot_phi_dofs, V)
    bc_phi_right = fem.dirichletbc(default_scalar_type(0.0), right_phi_dofs, V)
    bc_phi_left = fem.dirichletbc(default_scalar_type(0.0), left_phi_dofs, V)
    bc = [bc_bot_y, bc_bot_x, bc_top_y, bc_top_x]
    bc_phi = [bc_phi_top, bc_phi_bot]

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

A_ = crack_pattern[:, 0, :]/(2/domain_size)
B_ = crack_pattern[:, 1, :]/(2/domain_size)
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
solver_u.setTolerances(rtol=rtol, max_it=max_it)
solver_u.setFromOptions()

E_phi = (((l_0_**2) * ufl.dot(ufl.grad(p), ufl.grad(q))) + ((2*l_0_/G_c_) * H(u_new, H_old) +1 ) * p * q )* dx - (2*l_0_/G_c_) * H(u_new, H_old) * q * dx
a_phi = fem.form(ufl.lhs(E_phi))
L_phi = fem.form(ufl.rhs(E_phi))
A_phi = create_matrix(a_phi)
b_phi = create_vector(L_phi)

solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A_phi)
solver_phi.setType(ksp)
solver_phi.getPC().setType(pc)
solver_phi.setTolerances(rtol=rtol, max_it=max_it)
solver_phi.setFromOptions()

u_l2_error = fem.form(ufl.dot(u_new - u_old, u_new - u_old)*dx)
p_l2_error = fem.form(ufl.dot(p_new - p_old, p_new - p_old)*dx)

############################ new rxn force calculation ###############################################
B_bot = []
B_left = []
residual = ufl.action(ufl.lhs(E_du), u_new) - ufl.rhs(E_du)
v_reac = fem.Function(W)
virtual_work_form = fem.form(ufl.action(residual, v_reac))

bot_dofs = fem.locate_dofs_geometrical(W, bottom_boundary)
u_bc_bot = fem.Function(W)
bc_bot_rxn = fem.dirichletbc(u_bc_bot, bot_dofs)

left_dofs = fem.locate_dofs_geometrical(W, left_boundary)
u_bc_left = fem.Function(W)
bc_left_rxn = fem.dirichletbc(u_bc_left, left_dofs)
bc_rxn = [bc_bot_rxn, bc_left_rxn]

def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

if sim_case == "tension":
    u_bc_bot.sub(1).interpolate(one)
    u_bc_left.sub(0).interpolate(one)
else:
    u_bc_bot.sub(0).interpolate(one)

H_expr = fem.Expression(ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old), VV.element.interpolation_points())

################################# main simulation loop ###############################################
delta_T = delta_T1
error_tol = fem.Constant(domain, 1e-4)
error_total = fem.Constant(domain, 1.0)

reaction_forces_bot = np.zeros(num_steps+1)
if sim_case == "tension":
    reaction_forces_left = np.zeros(num_steps+1)

for i in range(num_steps+1):
    flag = 1
    staggered_iter = 0
    t_.value = delta_T.value * i
    u_bc_top.value = t_.value
    u_bc_right.value = t_.value
    error_total.value = 1.0
    
    while flag:
        staggered_iter +=1
        if error_total.value < error_tol.value:
            flag = 0
            break
            
        # Solve displacement problem
        A_u.zeroEntries()
        assemble_matrix(A_u, a_u, bcs=bc)
        A_u.assemble()
        with b_u.localForm() as loc:
            loc.set(0)
        assemble_vector(b_u, L_u)
        apply_lifting(b_u, [a_u], [bc])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_u, bc)
        solver_u.solve(b_u, u_new.vector)
        u_new.x.scatter_forward()

        # Solve phase-field problem
        A_phi.zeroEntries()
        assemble_matrix(A_phi, a_phi, bcs=bc_phi)
        A_phi.assemble()
        with b_phi.localForm() as loc:
            loc.set(0)
        assemble_vector(b_phi, L_phi)
        b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_phi, bc_phi)
        solver_phi.solve(b_phi, p_new.vector)
        p_new.x.scatter_forward()

        error_total.value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(u_l2_error), op=MPI.SUM)) + np.sqrt(domain.comm.allreduce(fem.assemble_scalar(p_l2_error), op=MPI.SUM))
        p_old.x.array[:] = p_new.x.array
        u_old.x.array[:] = u_new.x.array
        H_old.interpolate(H_expr)
        if rank == 0:
            print(f"step = {i}, error_total = {error_total.value}")
    ################################################################################
    if sim_case == "tension":
        v_reac.vector.set(0.0)
        v_reac.x.scatter_forward()
        fem.set_bc(v_reac.vector, [bc_bot_rxn])
        R_bot = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

        v_reac.vector.set(0.0)
        v_reac.x.scatter_forward()
        fem.set_bc(v_reac.vector, [bc_left_rxn])
        R_left = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)
    else:
        v_reac.vector.set(0.0)
        v_reac.x.scatter_forward()
        fem.set_bc(v_reac.vector, [bc_bot_rxn])
        R_bot = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

    if rank == 0:
        if sim_case == "tension":
            B_bot.append([np.sum(R_bot), i * delta_T.value])
            B_left.append([np.sum(R_left), i * delta_T.value])
        else:
            B_bot.append([np.sum(R_bot), i * delta_T.value])

    if i % save_freq == 0:
        out_file_name.write_function(p_new, i * delta_T.value)
        out_file_name_u.write_function(u_new, i * delta_T.value)
        
        plotter_func(p_new, dim=1, mesh = domain, title=f"{out_file}/p_{i}")
        if rank == 0:
            print(f"step = {i},  iter = {staggered_iter}, error total = {error_total.value}", flush=True)
            plot_force_disp(B_bot, "bot_rxn", out_file)
            if sim_case == "tension":
                plot_force_disp(B_left, "left_rxn", out_file)

end_time = time.time()
total_time = end_time - start_time

out_file_name.close()
out_file_name_u.close()
if rank == 0:
    print(f"Simulation completed in {total_time:.2f} seconds")
    print(f"rank = {rank} done.")
