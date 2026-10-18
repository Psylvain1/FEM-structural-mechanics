import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type, io
import ufl
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from utils import (
    BlockedLinearProblem,
    interface_int_entities,
    interpolate_submesh_to_parent,
)
import basix
from IPython.display import clear_output, HTML
from RCcolumn import * # Make sure this is in your directory
import vtk
import matplotlib.pyplot as plt
from dolfinx import geometry
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import os
from matplotlib import font_manager

# ==========================================
# 1. Paramètres du modèle
# ==========================================

# Propriétés Béton de coffrage (Gauche)
E1 = 30e6  # 38 GPa
nu1 = 0.2

# Propriétés Béton coulé (Droite)
E2 = 30e6  # 38 GPa
nu2 = 0.2

largeur, H, e_3DP, L, es, phi = create_geometry()
domain, cell_tags, facet_tags = io.gmshio.read_from_msh("colonne.msh", MPI.COMM_WORLD, 0, gdim=3)
tdim = domain.topology.dim
fdim = domain.topology.dim - 1

# Espaces pour définir les propriétés hétérogènes sur le maillage global
V_E = fem.functionspace(domain, ("DG", 0))  # FEniCSx 0.9.0
E_func = fem.Function(V_E)

cells_U = cell_tags.find(1)      # Tag 1 = Coffrage_U
cells_Noyau = cell_tags.find(2)  # Tag 2 = Coeur_Coule
cells_Aciers = cell_tags.find(3) # Tag 3 = Aciers

E_func.x.array[cells_U] = E1          # Béton 1
E_func.x.array[cells_Noyau] = E2      # Béton 2
E_func.x.array[cells_Aciers] = 210e9  # Acier

nu = 0.2
mu = E_func / (2.0 * (1.0 + nu))
lmbda = E_func * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(v):
    return ufl.sym(ufl.grad(v))

def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(tdim) + 2.0 * mu * epsilon(v)

# ==========================================
# 2. Création des Sous-Maillages (CG Mixte)
# ==========================================
INT_TAG = 4
LOAD_TAG = 3
INNER_FACET_TAG = 1

# Grouper les cellules par côté de l'interface
# Côté 1 : Coffrage U (1) seul
# Côté 2 : Noyau coulé (2) ET Aciers (3) parfaitement liés
cells_side1 = cell_tags.find(1)
cells_side2 = np.concatenate([cell_tags.find(2), cell_tags.find(3)])

subdomain1, sub1_cell_map, sub1_vertex_map, _ = dolfinx.mesh.create_submesh(domain, tdim, cells_side1)
subdomain2, sub2_cell_map, sub2_vertex_map, _ = dolfinx.mesh.create_submesh(domain, tdim, cells_side2)

# Interface (Tag 4)
facets_interface = facet_tags.find(INT_TAG)
interface_mesh, interface_cell_map, _, _ = dolfinx.mesh.create_submesh(domain, fdim, facets_interface)

# ==========================================
# PREVIEW PYVISTA : Vérification de l'interface CMZ
# ==========================================
if MPI.COMM_WORLD.rank == 0:
    import pyvista as pv
    from dolfinx import plot

    # Initialize PyVista Plotter
    plotter = pv.Plotter()

    # Create PyVista grids from FEniCSx meshes
    grid_sub1 = pv.UnstructuredGrid(*plot.vtk_mesh(subdomain1))
    grid_sub2 = pv.UnstructuredGrid(*plot.vtk_mesh(subdomain2))
    grid_interface = pv.UnstructuredGrid(*plot.vtk_mesh(interface_mesh))

    # Add Side 1 (Coffrage) - Highly transparent blue
    plotter.add_mesh(grid_sub1, color="lightblue", opacity=0.15, show_edges=False, label="Side 1: Coffrage")
    
    # Add Side 2 (Noyau + Aciers) - Semi-transparent gray
    plotter.add_mesh(grid_sub2, color="gray", opacity=0.3, show_edges=False, label="Side 2: Noyau & Aciers")
    
    # Add Interface (CMZ) - Bright red, solid, with edges to make it pop
    plotter.add_mesh(grid_interface, color="red", line_width=3, show_edges=True, label="Interface CMZ (Tag 4)")

    # Styling
    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background("white")
    
    # Optional: Set a good starting camera angle
    plotter.view_isometric()
    
    print("Affichage de l'interface en cours... Fermez la fenêtre PyVista pour lancer la résolution.")
    #plotter.show()
# ==========================================
# 3. Espaces de fonctions continus et bloc mixte
# ==========================================
# Espaces CG continus standards (Lagrange) sur chaque sous-domaine
V1 = fem.functionspace(subdomain1, ("CG", 1, (tdim,)))
V2 = fem.functionspace(subdomain2, ("CG", 1, (tdim,)))

W = ufl.MixedFunctionSpace(V1, V2)
u1 = fem.Function(V1, name="Disp_Side1")
u2 = fem.Function(V2, name="Disp_Side2")

v1, v2 = ufl.TestFunctions(W)
du1, du2 = ufl.TrialFunctions(W)

# Espace DG global UNIQUEMENT pour l'export Paraview
V_out = fem.functionspace(domain, ("DG", 1, (tdim,)))
u_out = fem.Function(V_out, name="Displacement")

# ==========================================
# 4. Entity Maps pour le solveur bloqué
# ==========================================
cell_imap = domain.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts

# Map pour le sous-domaine 1
domain_to_subdomain1 = np.full(num_cells, -1, dtype=np.int32)
domain_to_subdomain1[sub1_cell_map] = np.arange(len(sub1_cell_map), dtype=np.int32)

# Map pour le sous-domaine 2
domain_to_subdomain2 = np.full(num_cells, -1, dtype=np.int32)
domain_to_subdomain2[sub2_cell_map] = np.arange(len(sub2_cell_map), dtype=np.int32)

# Map pour l'interface (CORRECTION ICI)
facet_imap = domain.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
domain_to_interface = np.full(num_facets, -1, dtype=np.int32)
domain_to_interface[interface_cell_map] = np.arange(len(interface_cell_map), dtype=np.int32)

# Récupération des entités pour la mesure personnalisée
interface_entities, domain_to_subdomain1, domain_to_subdomain2 = interface_int_entities(
    domain, facets_interface, domain_to_subdomain1, domain_to_subdomain2
)

# Le dictionnaire vital pour l'assembleur
entity_maps = {
    interface_mesh: domain_to_interface,
    subdomain1: domain_to_subdomain1,
    subdomain2: domain_to_subdomain2,
}
# ==========================================
# 5. Mesures d'intégration
# ==========================================
dx_domain = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

def load_surface_locator(x):
    return np.isclose(x[2], L,atol=1e-5) 

# Trouver toutes les facettes chargées sur le maillage global parent
facets_load = mesh.locate_entities_boundary(domain, fdim, load_surface_locator)

# Créer la connectivité pour savoir quelles facettes appartiennent à quels volumes
domain.topology.create_connectivity(tdim, fdim)
domain.topology.create_connectivity(fdim, tdim)

# Identifier toutes les facettes du parent appartenant au Side 1 et Side 2
facets_side1 = mesh.compute_incident_entities(domain.topology, cells_side1, tdim, fdim)
facets_side2 = mesh.compute_incident_entities(domain.topology, cells_side2, tdim, fdim)

# Intersections : on sépare les facettes de charge selon le côté
facets_load_1 = np.intersect1d(facets_load, facets_side1)
facets_load_2 = np.intersect1d(facets_load, facets_side2)

# Nouveaux Tags pour séparer la charge
LOAD_TAG_1 = 31 # Charge sur le coffrage
LOAD_TAG_2 = 32 # Charge sur le noyau

# Combiner et trier pour créer le "meshtags" sur le domaine PARENT
all_load_facets = np.hstack([facets_load_1, facets_load_2])
all_load_values = np.hstack([
    np.full_like(facets_load_1, LOAD_TAG_1, dtype=np.int32), 
    np.full_like(facets_load_2, LOAD_TAG_2, dtype=np.int32)
])

sort_idx = np.argsort(all_load_facets)
facet_tags_exterior = mesh.meshtags(domain, fdim, all_load_facets[sort_idx], all_load_values[sort_idx])

# Mesure ds sur le domaine PARENT (Règle la ValueError)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_exterior)

dx_int = ufl.Measure("dx", domain=interface_mesh)

# Mesure d'interface personnalisée pour relier les deux sous-domaines
dInt = ufl.Measure("dS", domain=domain, subdomain_data=[(INT_TAG, interface_entities)], subdomain_id=INT_TAG)

Gamma = fem.assemble_scalar(fem.form(1 * dx_int))


# ==========================================
# 6. Loi de zone cohésive (CZM)
# ==========================================
V_int = fem.functionspace(interface_mesh, ("DG", 1))
d = fem.Function(V_int, name="Interfacial_damage")
d_prev = fem.Function(V_int, name="Previous_interfacial_damage")
d_old = fem.Function(V_int, name="Interfacial_damage_old")

#Propriétés du bulk en traction MODE I
Gc_nplot = 2550.0     # 6635 J/m²
sig_max_nplot = 1.5e6  # 1.5 MPa
K_penplot = 3.4e10     #1e9 # 10^13 N/m³

#Propriétés du bulk en cisaillement MODE II et MODE III
Gc_tplot = 4500     # 15000 J/m²
sig_max_tplot = 3e6  # 3.0 MPa
K_pentplot = 3.4e10  #1e9     # 10^10 N/m³

# Note: Using fem.Constant for simplicity on interface submesh properties to avoid mapping custom tags 
# unless you specifically need heterogeneous interface properties.
Gc_n = fem.Constant(interface_mesh, default_scalar_type(Gc_nplot))
sig_max_n = fem.Constant(interface_mesh, default_scalar_type(sig_max_nplot))
Gc_t = fem.Constant(interface_mesh, default_scalar_type(Gc_tplot))
sig_max_t = fem.Constant(interface_mesh, default_scalar_type(sig_max_tplot))

K_pen = fem.Constant(domain, K_penplot)
K_pent = fem.Constant(domain, K_pentplot)

delta_0_n = sig_max_n / K_pen
delta_0_t = sig_max_t / K_pent 

def normal_opening(v, n):
    return ufl.dot(v, n)

def tangential_opening(v, n):
    return v - normal_opening(v, n) * n

# Définition manuelle du saut entre Side 2 et Side 1
def jump(u_1, u_2):
    return u_2("-") - u_1("+")

n1_to_2 = ufl.FacetNormal(domain)("+")

def T(jump_v, jump_u, d_val, n):
    dn = normal_opening(jump_v, n)
    dt_vec = tangential_opening(jump_v, n)
    dn_u = normal_opening(jump_u, n)
    
    Kn = ufl.conditional(ufl.ge(dn_u, 0.0), K_pen * (1.0 - d_val), K_pen)
    Kt = K_pent * (1.0 - d_val) 
    
    return Kn * dn * n + Kt * dt_vec
# ==========================================
# 7. Forme Faible (Blocked)
# ==========================================
# Bulk: Intégration séparée sur le Side 1 (Tag 1) et Side 2 (Tags 2 et 3)
a_bulk = ufl.inner(sigma(du1), epsilon(v1)) * dx_domain(1) + \
         ufl.inner(sigma(du2), epsilon(v2)) * (dx_domain(2) + dx_domain(3))

# Interface: Lier du1 et du2
a_interface = ufl.dot(T(jump(du1, du2), jump(u1, u2), ufl.avg(d), n1_to_2), jump(v1, v2)) * dInt

a = a_bulk + a_interface

# 2. Constante d'intensité du couple
Force_surf = fem.Constant(domain, default_scalar_type(0.0)) 

# 3. Champ de traction tangentiel (Rotation autour de Z)
T_load = ufl.as_vector((
    0, 
    0,  
    Force_surf/(largeur*H)                
))
f = fem.Constant(domain, (0.0, 0, 0.0))

# Fext : Tout est intégré avec ds sur le parent, 
# mais en séparant physiquement les tags pour éviter le Segfault.
Fext = ufl.dot(f, v1) * dx_domain(1) + \
       ufl.dot(T_load, v1) * ds(LOAD_TAG_1) + \
       ufl.dot(f, v2) * (dx_domain(2) + dx_domain(3)) + \
       ufl.dot(T_load, v2) * ds(LOAD_TAG_2)

# Extraction des blocs pour le solveur
a_blocked_compiled = fem.form(ufl.extract_blocks(a), entity_maps=entity_maps)
Fext_blocked_compiled = fem.form(ufl.extract_blocks(Fext), entity_maps=entity_maps)

# ==========================================
# CZM: Calcul du dommage (Sur dInt)
# ==========================================
dn = normal_opening(jump(u1, u2), n1_to_2)
dt_vec = tangential_opening(jump(u1, u2), n1_to_2)

dn_plus = ufl.max_value(dn, 0.0) 
dt_sq = ufl.inner(dt_vec, dt_vec)

lambda_eff = ufl.sqrt((dn_plus / ufl.avg(delta_0_n))**2 + dt_sq / (ufl.avg(delta_0_t))**2)
lam_safe = ufl.max_value(lambda_eff, 1e-12)

lam_cr_n = ufl.avg(2.0 * Gc_n * K_pen / (sig_max_n**2))
lam_cr_t = ufl.avg(2.0 * Gc_t * K_pent / (sig_max_t**2))

w_n = (dn_plus / ufl.avg(delta_0_n))**2 / (lam_safe**2)
w_t = (dt_sq / (ufl.avg(delta_0_t))**2) / (lam_safe**2)

lam_cr = w_n * lam_cr_n + w_t * lam_cr_t
d_calc = (lam_cr / lam_safe) * (lambda_eff - 1.0) / (lam_cr - 1.0)

d_inst_borne = ufl.min_value(0.999, ufl.max_value(0.0, d_calc))
d_inst = ufl.conditional(ufl.ge(lambda_eff, 1.0), d_inst_borne, 0.0)
d_expr = ufl.max_value(ufl.avg(ufl.min_value(1.0, d_prev)), d_inst)

q_p = V_int.element.interpolation_points()
weights = np.full(q_p.shape[0], 0.5, dtype=dolfinx.default_scalar_type)
q_el = basix.ufl.quadrature_element(interface_mesh.basix_cell(), scheme="custom", points=q_p, weights=weights)
Q = fem.functionspace(interface_mesh, q_el)
q_ = ufl.TestFunction(Q)
dS_custom = ufl.Measure("dS", domain=domain, metadata={"quadrature_scheme": "custom", "quadrature_points": q_p, "quadrature_weights": weights}, subdomain_data=[(INT_TAG, interface_entities)], subdomain_id=INT_TAG)

facet_interp = fem.form(1 / ufl.FacetArea(domain) * d_expr * ufl.avg(q_) * dS_custom, entity_maps=entity_maps)

# ==========================================
# 8. Conditions aux Limites (Sur Side 1)
# ==========================================

# --- CORRECTION POUR LES SOUS-ESPACES ---
# On effondre (collapse) les sous-espaces pour lire leur géométrie
V0C, _ = V1.sub(0).collapse()
V1C, _ = V1.sub(1).collapse()
V2C, _ = V1.sub(2).collapse()

# 1. On passe le TUPLE (sous-espace_global, espace_effondré) pour faire le lien
right_dofs = fem.locate_dofs_geometrical((V1.sub(0), V0C), lambda x: np.logical_and.reduce([np.isclose(x[0], largeur/2,atol=3e-2),np.isclose(x[1], largeur/2,atol=3e-2),np.isclose(x[2], 0.0)]))
# Pareil pour V1
right_dofsB = fem.locate_dofs_geometrical((V1.sub(1), V1C), lambda x: np.logical_and.reduce([np.isclose(x[0], largeur/2,atol=3e-2),np.isclose(x[1], largeur/2,atol=3e-2),np.isclose(x[2], 0.0)]))
# Pareil pour V2
right_dofsC = fem.locate_dofs_geometrical((V1.sub(2), V2C), lambda x: np.isclose(x[2], 0.0))


# --- CORRECTION POUR LES SOUS-ESPACES ---
# On effondre (collapse) les sous-espaces pour lire leur géométrie
V0B, _ = V2.sub(0).collapse()
V1B, _ = V2.sub(1).collapse()
V2B, _ = V2.sub(2).collapse()

# 1. On passe le TUPLE (sous-espace_global, espace_effondré) pour faire le lien
right_dofsB = fem.locate_dofs_geometrical((V2.sub(0), V0B), lambda x: np.logical_and.reduce([np.isclose(x[0], largeur/2,atol=3e-2),np.isclose(x[1], largeur/2,atol=3e-2),np.isclose(x[2], 0.0)]))
# Pareil pour V1
right_dofsBB = fem.locate_dofs_geometrical((V2.sub(1), V1B), lambda x: np.logical_and.reduce([np.isclose(x[0], largeur/2,atol=3e-2),np.isclose(x[1], largeur/2,atol=3e-2),np.isclose(x[2], 0.0)]))
# Pareil pour V2
right_dofsCB = fem.locate_dofs_geometrical((V2.sub(2), V2B), lambda x: np.isclose(x[2], 0.0))


# --- Définition des conditions aux limites ---
bcs = [    
    # 2. On prend left_dofs[0] (les noeuds de V.sub(0)) et on applique sur V.sub(0)
    fem.dirichletbc(default_scalar_type(0.0), right_dofs[0], V1.sub(0)),
    fem.dirichletbc(default_scalar_type(0.0), right_dofsB[0], V1.sub(1)), 
    # Pareil pour l'axe Y
    fem.dirichletbc(default_scalar_type(0.0), right_dofsC[0], V1.sub(2)), 
    # 2. On prend left_dofs[0] (les noeuds de V.sub(0)) et on applique sur V.sub(0)
    fem.dirichletbc(default_scalar_type(0.0), right_dofsB[0], V2.sub(0)),
    fem.dirichletbc(default_scalar_type(0.0), right_dofsBB[0], V2.sub(1)), 
    # Pareil pour l'axe Y
    fem.dirichletbc(default_scalar_type(0.0), right_dofsCB[0], V2.sub(2)), 
]


# Résolveur bloqué
problem = BlockedLinearProblem(
    a_blocked_compiled,
    Fext_blocked_compiled,
    [u1, u2],
    bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    #petsc_options={"ksp_type": "cg", "pc_type": "bjacobi"} #"pc_type": "bjacobi" 
)


# ==========================================
# 9. Point de suivi & Boucle de résolution
# ==========================================
point_coord = np.array([[0.07, 0.05, 0.0]], dtype=default_scalar_type)
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, point_coord)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point_coord)

cells_for_point = []
points_on_proc = []
for idx, pt in enumerate(point_coord):
    if len(colliding_cells.links(idx)) > 0:
        points_on_proc.append(pt)
        cells_for_point.append(colliding_cells.links(idx)[0])

Disp_Y = [0.0]
Plot_f = [0.0]
Nincr = 10
loading = np.linspace(0, -2000000, Nincr + 1)

Niter_max = 200
tol = 1e-6
out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "a")
damage_file = io.VTKFile(MPI.COMM_WORLD, "results/damage.pvd", "a")

for i, t in enumerate(loading[1:]):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load step {i+1}/{len(loading)-1}")
        
    Force_surf.value = np.array(t, dtype=default_scalar_type)
    nRes = 1.0
    j = 0
    
    while j < Niter_max:
        problem.solve()
        d.x.array[:] = fem.assemble_vector(facet_interp).array
        nRes = np.sqrt((fem.assemble_scalar(fem.form((d - d_old) ** 2 * dx_int)))) / Gamma
        d_old.x.array[:] = d.x.array[:]
        j += 1
        print(f" Iteration {j:2d} | Residual: {nRes}")
        print("damage max", max(d.x.array[:]))
        if nRes < tol:
            break
    else:
        raise ValueError(f"Fixed-point solver did not converge in less than {Niter_max} iterations")
    
    d_prev.x.array[:] = d.x.array[:]
    

    # Mapping retour vers le parent pour l'export Paraview
    interpolate_submesh_to_parent(u_out, u1, sub1_cell_map)
    interpolate_submesh_to_parent(u_out, u2, sub2_cell_map)

    if len(points_on_proc) > 0:
        u_val = u_out.eval(points_on_proc, cells_for_point)
        uy_local = u_val[1] 
    else:
        uy_local = 0.0  

    uy_global = domain.comm.allreduce(uy_local, op=MPI.MIN)
    
    seuil_saut = 10000000000 
    saut_actuel = abs(uy_global - Disp_Y[-1])
    
    if saut_actuel > seuil_saut:
        if MPI.COMM_WORLD.rank == 0:
            print(f"⚠️ Arrêt du chargement : Saut de déplacement détecté ({saut_actuel:.2e} m > {seuil_saut} m).")
        
        Disp_Y.append(Disp_Y[-1])
        Plot_f.append(0)
        out_file.write_function(u_out, i)
        damage_file.write_function(d, i)
        break  

    Disp_Y.append(uy_global)
    Plot_f.append(t * 0.006)

    out_file.write_function(u_out, i)
    damage_file.write_function(d, i)
    clear_output(wait=True)
    

out_file.close()
damage_file.close()

# ==========================================
# Tracé
# ==========================================
if MPI.COMM_WORLD.rank == 0:
    plt.figure(figsize=(8, 6))
    plt.plot(np.abs(Disp_Y), np.abs(Plot_f), marker='.', linestyle='-', color='b')
    plt.xlabel("Déplacement au point (0.07, 0.05, 0.0) en Y (m)")
    plt.ylabel("Force appliquée (N)")
    plt.grid(True)
    plt.savefig("results/Force_Deplacement_Curve.pdf", bbox_inches='tight')
    #plt.show()