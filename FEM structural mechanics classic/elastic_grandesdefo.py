import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type, io
import dolfinx.plot as plot
import ufl
from dolfinx.fem.petsc import NonlinearProblem  # <-- Nouveau solveur
from dolfinx.nls.petsc import NewtonSolver      # <-- Nouveau solveur
from utils import *
import basix
from IPython.display import clear_output, HTML
from geometry_flambement import *
import vtk
import pyvista as pv
import matplotlib.pyplot as plt
from petsc4py import PETSc

L, H, e = 2.0, 0.2, 0.2  # Dimensions de la poutre (m)
x_interface = L / 2.0  # Position de l'interface

# Propriétés Béton 1 du coffrage
E1 = 36e9  # 36 GPa
nu1 = 0.2

# Propriétés Béton 2 de remplissage
E2 = 38e9  # 38 GPa
nu2 = 0.2

# ==========================================
# 2. Maillage et définition des sous-domaines
# ==========================================
# (Remplacé par l'import Gmsh)
largeur, H, e_3DP, L, es, phi = create_geometry()
domain, cell_tags, facet_tags = io.gmshio.read_from_msh("poutre.msh", MPI.COMM_WORLD, 0, gdim=3)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

def interface_locator(x):
    return np.isclose(x[2], x_interface, atol=1e-5)

V_E = fem.functionspace(domain, ("DG", 0))
E_func = fem.Function(V_E)

cells_U = cell_tags.find(1)      # Tag 1 = Coffrage_U
cells_Noyau = cell_tags.find(2)  # Tag 2 = Coeur_Coule
cells_Aciers = cell_tags.find(3) # Tag 3 = Aciers

# ==========================================
# 4. Assignation des propriétés matérielles
# ==========================================
E_func.x.array[cells_U] = E1          
E_func.x.array[cells_Noyau] = E2      
E_func.x.array[cells_Aciers] = 210e9  

nu = 0.2
mu = E_func / (2.0 * (1.0 + nu))
lmbda = E_func * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

tdim = domain.topology.dim
V = fem.functionspace(domain, ("CG", 1, (tdim,)))
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V) # Utilisé pour le Jacobien
# Integration in domain
dx = ufl.Measure("dx", domain=domain)


# =====================================================================
# ### MODIFICATION FLAMBEMENT 1 : Cinématique des grandes déformations
# =====================================================================
I = ufl.variable(ufl.Identity(tdim))
F_def = I + ufl.grad(u)             # Gradient de déformation
C = F_def.T * F_def                 # Tenseur de Cauchy-Green droit
E_GL = 0.5 * (C - I)                # Tenseur de déformation de Green-Lagrange

# Loi de comportement (Second tenseur de Piola-Kirchhoff)
S = lmbda * ufl.tr(E_GL) * I + 2.0 * mu * E_GL

# Premier tenseur de Piola-Kirchhoff (nécessaire pour l'équilibre spatial)
P = F_def * S

# Forme faible (Énergie interne)
a_bulk = ufl.inner(P, ufl.grad(v)) * dx
# =====================================================================

# Interface tags
fdim = domain.topology.dim - 1
INNER_FACET_TAG = 1 
INT_TAG = 2   

facets_interface = mesh.locate_entities(domain, fdim, interface_locator)
facet_map = domain.topology.index_map(fdim)
num_facets = facet_map.size_local + facet_map.num_ghosts
all_facets = np.arange(num_facets, dtype=np.int32)

facet_values = np.full(num_facets, INNER_FACET_TAG, dtype=np.int32)
facet_values[facets_interface] = INT_TAG
facet_tags_domain = mesh.meshtags(domain, fdim, all_facets, facet_values)

internal_facets = np.concatenate([facet_tags_domain.find(m) for m in [INNER_FACET_TAG, INT_TAG]])
interface_mesh, interfaces_cell_map, interfaces_vertex_map, _ = dolfinx.mesh.create_submesh(domain, fdim, internal_facets)

num_sub_facets = len(interfaces_cell_map)
interfaces_facet_tags = dolfinx.mesh.meshtags(
    interface_mesh,
    fdim,
    np.arange(num_sub_facets, dtype=np.int32),
    facet_tags_domain.values[interfaces_cell_map],
)

# Load Tags
LOAD_TAG = 3
def load_surface_locator(x):
    return np.isclose(x[2], L, atol=1e-5) 

facets_load = mesh.locate_entities_boundary(domain, fdim, load_surface_locator)
load_values = np.full_like(facets_load, LOAD_TAG, dtype=np.int32)
facet_tags_exterior = mesh.meshtags(domain, fdim, facets_load, load_values)

# Integration measures
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_exterior)


# =====================================================================
# ### MODIFICATION FLAMBEMENT 2 : Force axiale et Micro-Imperfection
# =====================================================================
Aire = largeur * H

# Force axiale (Paramètre de charge principal qui va augmenter)
Force_totale = fem.Constant(domain, default_scalar_type(0.0)) 

# Perturbation (10 Newtons transversaux pour initier le flambement)
Force_perturbation = fem.Constant(domain, default_scalar_type(10.0))

# Vecteur de traction (Tension de surface). Compression = Z négatif
T_load = ufl.as_vector((0, 0.0, Force_totale / Aire))

# Terme source externe
Fext = ufl.dot(T_load, v) * ds(LOAD_TAG)

# Résidu du système non-linéaire (Interne - Externe = 0)
F_form = a_bulk - Fext

# Calcul analytique du Jacobien pour la méthode de Newton
J_jacobian = ufl.derivative(F_form, u, du)
# =====================================================================

# Conditions aux limites
left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
bcs = [fem.dirichletbc(np.zeros((tdim,)), left_dofs, V)]

problem = NonlinearProblem(F_form, u, bcs=bcs, J=J_jacobian)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.max_it = 200

# --- NOUVEAU : Configuration du solveur linéaire (KSP) interne ---
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()

# Utilisation du Gradient Conjugué (très rapide pour les matrices symétriques)
opts[f"{option_prefix}ksp_type"] = "cg"
# Préconditionneur Multigrille (Magique pour la mécanique 3D)
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}ksp_rtol"] = 1e-5

ksp.setFromOptions()

# Loop résolution
Nincr = 10
# On pousse plus fort (ex: 2.5 MN) pour être sûr de dépasser la charge critique d'Euler
loading = np.linspace(0, -360000, Nincr + 1)

out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "a")

V_scalar = fem.functionspace(domain, ("DG", 0))

# =====================================================================
# ### MODIFICATION FLAMBEMENT 4 : Extraction des vraies contraintes
# En grandes déformations, on re-calcule les contraintes de Cauchy
# Sigma = (1 / det(F)) * P * F^T
# =====================================================================
J_det = ufl.det(F_def)
Cauchy_stress = (1.0 / J_det) * P * F_def.T

components = {
    "sigma_xx": (0, 0),
    "sigma_yy": (1, 1),
    "sigma_zz": (2, 2),
    "tau_xy": (0, 1), 
    "tau_xz": (0, 2), 
    "tau_yz": (1, 2)  
}

list_stress_funcs = []
list_stress_exprs = []

for name, (row, col) in components.items():
    func = fem.Function(V_scalar, name=name)
    # On cible la matrice des contraintes de Cauchy
    expr = fem.Expression(Cauchy_stress[row, col], V_scalar.element.interpolation_points())
    list_stress_funcs.append(func)
    list_stress_exprs.append(expr)

stress_file = io.VTKFile(MPI.COMM_WORLD, "results/stress.pvd", "a")

# PREPARATION POUR L'EXTRACTION
point_eval = np.array([[largeur - e_3DP, H / 2.0, L - 1e-6]], dtype=np.float64)
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point_eval)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, point_eval)

points_on_proc = []
cells_for_eval = []

if len(colliding_cells.links(0)) > 0:
    points_on_proc.append(point_eval[0])
    cells_for_eval.append(colliding_cells.links(0)[0])

load_history = [0.0] 
tau_zy_history = [0.0]
idx_tau_yz = list(components.keys()).index("sigma_xx") # Gardé comme dans ton code

# INITIALISATION DE L'ANIMATION PYVISTA
if MPI.COMM_WORLD.rank == 0:
    topology, cell_types, x_initial = plot.vtk_mesh(V)
    mesh_anim = pv.UnstructuredGrid(topology, cell_types, x_initial)
    plotter_anim = pv.Plotter(notebook=False, off_screen=True, window_size=[1920, 1080])
    
    mesh_anim.cell_data[r"$\sigma_{x,x}$"] = np.zeros(mesh_anim.n_cells)
    clim_max = 2e7 
    
    sargs = {
        "title": "",
        "label_font_size": 32,
        "position_x": 0.6,
        "height": 0.1,
        "width": 0.3
    }

    plotter_anim.add_text(
        r"$\sigma_{x,x}$", 
        position=(0.75, 0.15), 
        viewport=True, 
        font_size=32,
        name="tau_title" 
    )

    actor_anim = plotter_anim.add_mesh(
        mesh_anim, 
        scalars=r"$\sigma_{x,x}$", 
        cmap="jet", 
        show_edges=True,
        clim=[-clim_max, clim_max], 
        opacity=1,
        scalar_bar_args=sargs
    )
    
    plotter_anim.camera_position = [
        (0.4, 0.1, -1),
        (0.1, 0.1, 1.0),
        (0.0, 1.0, 0.0)
    ]   
    plotter_anim.open_gif("results/flambement_poutre.gif")

# =====================================================================
# BOUCLE TEMPORELLE
# =====================================================================
for i, t in enumerate(loading[1:]):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load step {i+1}/{len(loading)-1} - Charge = {t} N")
    
    # Mise à jour de la constante de charge
    Force_totale.value = np.array(t, dtype=default_scalar_type)
    
    # On résout avec le solveur de Newton
    n_iter, converged = solver.solve(u)
    if MPI.COMM_WORLD.rank == 0:
        print(f" -> Convergé en {n_iter} itérations de Newton.")

    out_file.write_function(u, i)

    for func, expr in zip(list_stress_funcs, list_stress_exprs):
        func.interpolate(expr)
        
    stress_file.write_function(list_stress_funcs, i)
    
    tau_func = list_stress_funcs[idx_tau_yz]
    
    if len(points_on_proc) > 0:
        val_locale = tau_func.eval(points_on_proc, cells_for_eval)[0]
    else:
        val_locale = 0.0
        
    val_globale = MPI.COMM_WORLD.allreduce(val_locale, op=MPI.SUM)
    
    if MPI.COMM_WORLD.rank == 0:
        load_history.append(t)
        tau_zy_history.append(val_globale)

    # CAPTURE DE L'IMAGE POUR L'ANIMATION PYVISTA
    if MPI.COMM_WORLD.rank == 0:
        disp_array = u.x.array.reshape((-1, domain.geometry.dim))
        
        # Amplification : réduire à 1.0 ou 5.0 en grandes déformations pour éviter les distorsions folles
        amplification = 50.0 
        
        mesh_anim.points[:] = x_initial + disp_array * amplification
        tau_array = tau_func.x.array[:mesh_anim.n_cells]
        mesh_anim.cell_data[r"$\sigma_{x,x}$"] = tau_array
        
        plotter_anim.add_text(
            f"Pas {i+1}/{len(loading)-1} | Force Axiale = {t:.0f} N", 
            font_size=12, 
            name="status_text"
        )        
        
        plotter_anim.render()
        plotter_anim.write_frame()

    clear_output(wait=True)

out_file.close()
stress_file.close()

# Export des matériaux
mat_func = fem.Function(V_E, name="Material_ID")
mat_func.x.array[cells_U] = 1.0 
mat_func.x.array[cells_Noyau] = 2.0 
mat_func.x.array[cells_Aciers] = 3.0 

with io.VTKFile(MPI.COMM_WORLD, "results/materials.pvd", "w") as mat_file:
    mat_file.write_function(mat_func, 0.0)

with io.XDMFFile(domain.comm, "results/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags_domain, domain.geometry)

if MPI.COMM_WORLD.rank == 0:
    frames_to_hold = 20
    for _ in range(frames_to_hold):
        plotter_anim.write_frame()

    plotter_anim.close()
    del plotter_anim 
    print("L'animation 3D a été enregistrée avec succès dans 'results/flambement_poutre.gif'")

# TRACÉ DE LA COURBE
if MPI.COMM_WORLD.rank == 0:
    plt.figure(figsize=(8, 6))
    # graphics_setup() # Décommente si tu as cette fonction dans utils.py
    
    loads_plot = np.abs(load_history)
    plt.plot(loads_plot, tau_zy_history, marker='o', linestyle='-', color='#d62728', linewidth=2)
    plt.xlabel("Force Axiale (N)", fontsize=12)
    plt.ylabel(r"Contrainte cible", fontsize=12)
    plt.title("Suivi contrainte vs. Charge Axiale", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/courbe_flambement.pdf", dpi=300)