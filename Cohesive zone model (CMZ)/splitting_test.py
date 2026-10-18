import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type, io
import ufl
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from utils import *
import basix
from IPython.display import clear_output, HTML

# ==========================================
# 1. Paramètres du modèle
# ==========================================
L, H, e = 0.1, 0.1, 0.2  # Dimensions de la poutre (m)
x_interface = 0.04  # Position de l'interface
e_3DP=0.04 #Epaisseur du coffrage imprimé
# Propriétés Béton 1 (Gauche)
E1 = 38e9  # 38 GPa
nu1 = 0.2

# Propriétés Béton 2 (Droite)
E2 = 38e9  # 38 GPa
nu2 = 0.2

# ==========================================
# 2. Maillage et définition des sous-domaines
# ==========================================
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, H, e])],
    [10, 2, 10],
    cell_type=mesh.CellType.tetrahedron,
)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))


def interface_locator(x):
    Left =np.isclose(x[0],e_3DP,atol=1e-6)
    return Left    #np.isclose(x[1], x_interface, atol=1e-6)

V_E = fem.functionspace(domain, ("DG", 0))
E_func = fem.Function(V_E)

cells_left = mesh.locate_entities(domain, domain.topology.dim, interface_locator)
E_func.x.array[:] = E1
E_func.x.array[cells_left] = E2

nu = 0.2
mu = E_func / (2.0 * (1.0 + nu))
lmbda = E_func * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(v):
    return ufl.sym(ufl.grad(v))

def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(tdim) + 2.0 * mu * epsilon(v)

tdim = domain.topology.dim
V = fem.functionspace(domain, ("DG", 1, (tdim,)))
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

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
interface_mesh, interfaces_cell_map, interfaces_vertex_map, _ = (
    dolfinx.mesh.create_submesh(domain, fdim, internal_facets)
)

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
    return np.isclose(x[0], L)

facets_load = mesh.locate_entities_boundary(domain, fdim, load_surface_locator)
load_values = np.full_like(facets_load, LOAD_TAG, dtype=np.int32)
facet_tags_exterior = mesh.meshtags(domain, fdim, facets_load, load_values)

# Integration
dx = ufl.Measure("dx", domain=domain)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags_domain, subdomain_id=1)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_exterior)
dx_int = ufl.Measure("dx", domain=interface_mesh)

Gamma_int = fem.assemble_scalar(fem.form(1 * dS(INT_TAG)))
Gamma_bulk = fem.assemble_scalar(fem.form(1 * dS(INNER_FACET_TAG)))
Gamma = fem.assemble_scalar(fem.form(1 * dx_int))
print(f"Total interface area = {Gamma_int:.3f}")

V_int = fem.functionspace(interface_mesh, ("DG", 1))
d = fem.Function(V_int, name="Interfacial_damage")
d_prev = fem.Function(V_int, name="Previous_interfacial_damage")
d_old = fem.Function(V_int, name="Interfacial_damage_old")

# ==========================================
# CZM: LOI BILINÉAIRE - Paramètres
# ==========================================

#Propriétés du bulk en traction MODE I
Gc_nplot = 2550.0     # 6635 J/m²
sig_max_nplot = 2.5e6  # 1.5 MPa
K_penplot = 3.4e10     #1e9 # 10^13 N/m³

#Propriétés du bulk en cisaillement MODE II et MODE III
Gc_tplot = 4500     # 15000 J/m²
sig_max_tplot = 3e6  # 3.0 MPa
K_pentplot = 3.4e10  #1e9     # 10^10 N/m³
# --- Mode I (Traction) ---
# Objectif : sigma = 3 MPa, delta_0 = 1 mm, delta_cr = 2 mm
Gc_n = create_piecewise_constant_field(interface_mesh, facet_tags_domain, {INT_TAG: 1170}, name="Gc_n", default_value=Gc_nplot) #500.0
sig_max_n = create_piecewise_constant_field(interface_mesh, facet_tags_domain, {INT_TAG: 1.3e6}, name="sigmax_n", default_value=sig_max_nplot)

# --- Mode II (Cisaillement) ---
# Objectif : sigma = 1.5 MPa,  delta_cr = 2 mm Gc=1500
Gc_t = create_piecewise_constant_field(interface_mesh, facet_tags_domain, {INT_TAG: 1500}, name="Gc_t", default_value=Gc_tplot) #1170
sig_max_t = create_piecewise_constant_field(interface_mesh, facet_tags_domain, {INT_TAG: 1e6}, name="sigmax_t", default_value=sig_max_tplot)

# Raideur de pénalité explicite (très grande pour fermer la fissure élastiquement)
K_pen = fem.Constant(domain,K_penplot)
K_pent = fem.Constant(domain,K_pentplot)

# Déplacements d'initiation (fin de la phase élastique)
delta_0_n = sig_max_n / K_pen
delta_0_t = sig_max_t / K_pent  # <-- Correction de la typo K_pentr

def normal_opening(v, n):
    return ufl.dot(v, n)

def tangential_opening(v, n):
    return v - normal_opening(v, n) * n

def normalized_effective_opening(v, n):
    dn = normal_opening(v, n)
    dt_vec = tangential_opening(v, n)
    dn_plus = ufl.max_value(dn, 0.0) 
    dt_sq = ufl.inner(dt_vec, dt_vec)
    return ufl.sqrt((dn_plus / ufl.avg(delta_0_n))**2 + dt_sq / (ufl.avg(delta_0_t))**2)

def T(jump_v, jump_u, d, n):
    dn = normal_opening(jump_v, n)
    dt_vec = tangential_opening(jump_v, n)
    dn_u = normal_opening(jump_u, n)
    
    # La raideur pénalisée est appliquée ou dégradée
    Kn = ufl.conditional(ufl.ge(dn_u, 0.0), K_pen * (1.0 - d), K_pen)
    Kt = K_pent * (1.0 - d)  # <-- Utilisation de la raideur tangentielle
    
    return Kn * dn * n + Kt * dt_vec

# Weak form formulation
n_facet = ufl.FacetNormal(domain)
a_bulk = ufl.inner(sigma(du), epsilon(v)) * dx
a_interface = ufl.dot(T(ufl.jump(du), ufl.jump(u), ufl.avg(d), n_facet("-")), ufl.jump(v)) * dS(INT_TAG) \
            + ufl.dot(T(ufl.jump(du), ufl.jump(u), ufl.avg(d), n_facet("-")), ufl.jump(v)) * dS(INNER_FACET_TAG)
a = a_bulk + a_interface

T_load = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
f = fem.Constant(domain, (0.0, 0, 0.0))
Fext = ufl.dot(f, v) * dx + ufl.dot(T_load, v) * ds(LOAD_TAG)

facet_imap = domain.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
domain_to_interfaces = np.full(num_facets, 1, dtype=np.int32) 
domain_to_interfaces[interfaces_cell_map] = np.arange(len(interfaces_cell_map), dtype=np.int32)
entity_maps = {interface_mesh: domain_to_interfaces}

a_compiled = fem.form(a, entity_maps=entity_maps)
Fext_compiled = fem.form(Fext, entity_maps=entity_maps)

#mesh_preview(domain, fdim, facets_interface)


# ==========================================
# CZM: Calcul du dommage Bilinéaire MIXTE
# ==========================================
# On recrée les composantes d'ouverture pour calculer les poids
dn = normal_opening(ufl.jump(u), n_facet("-"))
dt_vec = tangential_opening(ufl.jump(u), n_facet("-"))

dn_plus = ufl.max_value(dn, 0.0) 
dt_sq = ufl.inner(dt_vec, dt_vec)

# 1. Ouverture effective normalisée
lambda_eff = ufl.sqrt(
    (dn_plus / ufl.avg(delta_0_n))**2 +
    dt_sq / (ufl.avg(delta_0_t))**2)

lam_safe = ufl.max_value(lambda_eff, 1e-12)

# 2. Variable instantanée 
lambda_max = lambda_eff  

# 3. Points critiques purs
lam_cr_n = ufl.avg(2.0 * Gc_n * K_pen / (sig_max_n**2))
lam_cr_t = ufl.avg(2.0 * Gc_t * K_pent / (sig_max_t**2))

# 4. Poids de mixité (normalisation correcte)
w_n = (dn_plus / ufl.avg(delta_0_n))**2 / (lam_safe**2)
w_t = (dt_sq / (ufl.avg(delta_0_t))**2) / (lam_safe**2)

# 5. Paramètre critique mixte
lam_cr = w_n * lam_cr_n + w_t * lam_cr_t

# 6. Loi bilinéaire CORRECTE (Ajout de la fraction inverse pour la ligne droite)
d_calc = (lam_cr / lam_safe) * (lambda_max - 1.0) / (lam_cr - 1.0)

# 7. Bornage (Empêche d d'être négatif avant le pic ou de dépasser 1 après la rupture)
d_inst_borne = ufl.min_value(
    0.999,
    ufl.max_value(0.0, d_calc)
)

# 8. Activation (Sécurité absolue : 0 strict tant que le pic n'est pas atteint)
d_inst = ufl.conditional(
    ufl.ge(lambda_max, 1.0),
    d_inst_borne,
    0.0
)

# 9. Irréversibilité (La Fissure ne se répare pas en cas de décharge)
d_expr = ufl.max_value(ufl.avg(ufl.min_value(1.0, d_prev)), d_inst)

# ==========================================

q_p = V_int.element.interpolation_points()
weights = np.full(q_p.shape[0], 1.0, dtype=dolfinx.default_scalar_type)
q_el = basix.ufl.quadrature_element(interface_mesh.basix_cell(), scheme="custom", points=q_p, weights=weights)
Q = fem.functionspace(interface_mesh, q_el)
q_ = ufl.TestFunction(Q)
dS_custom = ufl.Measure("dS", domain=domain, metadata={"quadrature_scheme": "custom", "quadrature_points": q_p, "quadrature_weights": weights}, subdomain_data=facet_tags_domain)

facet_interp = fem.form(1 /(2*ufl.FacetArea(domain)) * d_expr * ufl.avg(q_) * dS_custom, entity_maps=entity_maps)

# Boundary conditions
left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))

bcs = [
    fem.dirichletbc(np.zeros((tdim,)), left_dofs, V)
]

v_reac = fem.Function(V)
fem.set_bc(v_reac.x.array, bcs)
virtual_work_form = fem.form(ufl.action(ufl.action(a, u) - Fext, v_reac), entity_maps=entity_maps)

# Problem
problem = LinearProblem(
    a_compiled,
    Fext_compiled,
    u=u,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"} #petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
)

interface_damage = fem.form(ufl.avg(d) * dS(INT_TAG), entity_maps=entity_maps)
bulk_damage = fem.form(ufl.avg(d) * dS(INNER_FACET_TAG), entity_maps=entity_maps)



# Ajoutez ces imports au début avec les autres
from dolfinx import geometry
import matplotlib.pyplot as plt
# ==========================================
# Ajout : Préparation pour le suivi du déplacement au point (0.07, 0.05, 0.0)
# ==========================================
point_coord = np.array([[0.04, 0.05, 0.0]], dtype=default_scalar_type)

# Recherche de la cellule contenant le point
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, point_coord)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point_coord)

cells_for_point = []
points_on_proc = []
for idx, pt in enumerate(point_coord):
    if len(colliding_cells.links(idx)) > 0:
        points_on_proc.append(pt)
        cells_for_point.append(colliding_cells.links(idx)[0])

# Initialisation de la liste pour stocker le déplacement vertical (Y)
Disp_Y = [0.0]
Plot_f=[0.0]

# Loop résolution
Nincr = 60



loading = np.linspace(0, 1500000, Nincr + 1) #Force surfacique de 1.5 MPa (30 kN)

Niter_max = 200
tol = 1e-5
damage_results = [[0.0, 0.0, 0.0]]
Force = [0.0]
iterations = [0]
out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "a")
damage_file = io.VTKFile(MPI.COMM_WORLD, "results/damage.pvd", "a")


# ==========================================
# Déclenchement du tracé avec les VRAIES valeurs
# ==========================================





#plot_czm_laws(Gc_nplot, sig_max_nplot, K_penplot, Gc_tplot, sig_max_tplot, K_pentplot)

for i, t in enumerate(loading[1:]):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load step {i+1}/{len(loading)-1}")
    T_load.value[2] = t
    nRes = 1.0
    j = 0
    while j < Niter_max:
        problem.solve()
        d.x.array[:] = fem.assemble_vector(facet_interp).array
        nRes = np.sqrt((fem.assemble_scalar(fem.form((d - d_old) ** 2 * dx_int)))) / Gamma
        d_old.x.array[:] = d.x.array[:]
        j += 1
        print(f" Iteration {j:2d} | Residual: {nRes}")
        if nRes < tol:
            break
    else:
        raise ValueError(f"Fixed-point solver did not converge in less than {Niter_max} iterations")
    
    iterations.append(j)
    d_prev.x.array[:] = d.x.array[:]
    print("damage max", max(d.x.array[:]))
    print("labmda_eff",fem.assemble_scalar(fem.form(lambda_eff*dS(INT_TAG), entity_maps=entity_maps)))
    print("w_n",fem.assemble_scalar(fem.form(w_n*dS(INT_TAG), entity_maps=entity_maps)))
    print("w_t",fem.assemble_scalar(fem.form(w_t*dS(INT_TAG), entity_maps=entity_maps)))
    print("lam_cr",fem.assemble_scalar(fem.form(lam_cr*dS(INT_TAG), entity_maps=entity_maps)))
    print("lamb_cr_t",fem.assemble_scalar(fem.form(lam_cr_t*dS(INT_TAG), entity_maps=entity_maps)))
    print("d_calc",fem.assemble_scalar(fem.form(d_calc * dS(INT_TAG), entity_maps=entity_maps)))
    print("d_int_borne",fem.assemble_scalar(fem.form(d_inst_borne * dS(INT_TAG), entity_maps=entity_maps)))
    print("d_inst",fem.assemble_scalar(fem.form(d_inst* dS(INT_TAG), entity_maps=entity_maps)))
    print("d_exp",fem.assemble_scalar(fem.form(d_expr* dS(INT_TAG), entity_maps=entity_maps)))
    print("d",max(d.x.array[:]))
    Force.append(fem.assemble_scalar(virtual_work_form))
    damage_results.append([fem.assemble_scalar(interface_damage), fem.assemble_scalar(bulk_damage)])

    # ==========================================
    # Ajout : Extraction du déplacement à chaque pas
    # ==========================================
    if len(points_on_proc) > 0:
        u_val = u.eval(points_on_proc, cells_for_point)
        uy_local = u_val[2]  # Indice 1 pour la composante z
    else:
        uy_local = 0.0  # Pour les cœurs MPI qui n'ont pas le point

    
    # Rassemblement de la valeur sur tous les cœurs (utilisation de MIN car on s'attend à un déplacement négatif)
    # Rassemblement de la valeur sur tous les cœurs (utilisation de MIN car on s'attend à un déplacement négatif)
    uy_global = domain.comm.allreduce(uy_local, op=MPI.MIN)
    
    # ==========================================
    # AJOUT : Critère d'arrêt sur le saut de déplacement
    # ==========================================
    seuil_saut = 1e-3  # Seuil toléré (ex: 0.1 mm). À ajuster selon votre besoin !
    
    # On calcule la différence absolue avec le déplacement précédent
    saut_actuel = abs(uy_global - Disp_Y[-1])
    if saut_actuel > seuil_saut:
        if MPI.COMM_WORLD.rank == 0:
            print(f"⚠️ Arrêt du chargement : Saut de déplacement détecté ({saut_actuel:.2e} m > {seuil_saut} m). Rupture probable.")
        
        # On ajoute quand même les dernières valeurs pour le graphique avant de quitter
        Disp_Y.append(Disp_Y[-1])
        Plot_f.append(0)
        
        # On sauvegarde le dernier état avant de casser la boucle
        out_file.write_function(u, i)
        damage_file.write_function(d, i)
        
        break  # <--- CASSE LA BOUCLE DE CHARGEMENT PRINCIPALE
    # ==========================================

    Disp_Y.append(uy_global)
    Plot_f.append(t*0.1*0.2)
    # ==========================================

    out_file.write_function(u, i)
    damage_file.write_function(d, i)
    clear_output(wait=True)

out_file.close()
damage_file.close()

with io.XDMFFile(domain.comm, "results/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags_domain, domain.geometry)


# ==========================================
# Ajout : Tracé de la courbe Force - Déplacement
# ==========================================
if MPI.COMM_WORLD.rank == 0:
    plt.figure(figsize=(8, 6))
    # On trace la valeur absolue pour avoir une belle courbe dans les axes positifs
    plt.plot(np.abs(Disp_Y), np.abs(Plot_f), marker='.', linestyle='-', color='b')
    plt.xlabel("Déplacement au point (0.04, 0.05, 0.0) en Z (m)")
    plt.ylabel("Force appliquée (N)")
    plt.title("Comportement Global : Force vs Déplacement")
    plt.grid(True)
    
    # Sauvegarde de l'image
    plt.savefig("results/Force_Deplacement_Curve.png")
    
    # Affichage
    plt.show()