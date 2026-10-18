import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type,io
import ufl
from dolfinx.fem.petsc import NonlinearProblem,LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from utils import *
import basix
from IPython.display import clear_output, HTML
# ==========================================
# 1. Paramètres du modèle
# ==========================================
L, H, e = 2.0, 0.2, 0.2  # Dimensions de la poutre (m)
x_interface = L / 2.0  # Position de l'interface

# Propriétés Béton 1 (Gauche)
E1 = 38e9  # 36 GPa
nu1 = 0.2

# Propriétés Béton 2 (Droite)
E2 = 38e9  # 38 GPa
nu2 = 0.2

# Propriétés de l'interface (Mohr-Coulomb)
# ==========================================
# 2. Maillage et définition des sous-domaines
# ==========================================
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, H, e])],
    [10, 10, 2],
    cell_type=mesh.CellType.tetrahedron,
)

# ---------------------------------------------------------
# CORRECTION FENICSX 0.9.0 : functionspace unifié
# On précise (domain.geometry.dim,) pour en faire un vecteur
# ---------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# Création des tags pour les cellules (matériaux)
# Interface localisation:
def interface_locator(x):
    return np.isclose(x[0], x_interface, atol=1e-6) #np.logical_and.reduce([np.isclose(x[1], x_interface), x[0] >= 0.0, x[0] <= L/2.0])

# Espaces pour définir les propriétés hétérogènes (DG0)
V_E = fem.functionspace(domain, ("DG", 0))  # FEniCSx 0.9.0
E_func = fem.Function(V_E)

# Assignation de E selon la position
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

# Weak form with a discontinuous displacement.
tdim = domain.topology.dim
V = fem.functionspace(domain, ("DG", 1, (tdim,)))
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# Création d'un meshtag pour restreindre l'intégrale dS à l'interface
# (Nécessaire dans les versions récentes pour cibler précisément la discontinuité)
fdim = domain.topology.dim - 1
# Interface tags :
INNER_FACET_TAG = 1 # Mesh dans la matrice
INT_TAG = 2   # Interface entre les deux matériaux

# Interface entities in domain :
facets_interface = mesh.locate_entities(domain, fdim, interface_locator)
facet_map = domain.topology.index_map(fdim)
num_facets = facet_map.size_local + facet_map.num_ghosts
all_facets = np.arange(num_facets, dtype=np.int32)
# 3. Create a values array filled entirely with 0 (INNER_FACET_TAG)
facet_values = np.full(num_facets, INNER_FACET_TAG, dtype=np.int32)
# 4. Overwrite the values with 1 ONLY where the interface is
facet_values[facets_interface] = INT_TAG
# 5. Build the meshtags passing ALL facets and ALL values
facet_tags_domain = mesh.meshtags(domain, fdim, all_facets, facet_values)

# Interface submesh :
#interface_mesh, entity_map = mesh.create_submesh(domain, fdim, facets_interface)[:2]
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

#Load Tags :

# --- NOUVEAU : Marqueur pour la charge surfacique ---
LOAD_TAG = 3

# Par exemple, on charge une zone de 20 cm au centre, sur la face supérieure (H)
def load_surface_locator(x):
    return np.isclose(x[1], H)

# Trouver les facettes frontières (extérieures) correspondant à cette zone
facets_load = mesh.locate_entities_boundary(domain, fdim, load_surface_locator)
load_values = np.full_like(facets_load, LOAD_TAG, dtype=np.int32)

# Créer un tag spécifique pour les bords extérieurs
facet_tags_exterior = mesh.meshtags(domain, fdim, facets_load, load_values)


# Integration in domain
dx = ufl.Measure("dx", domain=domain)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags_domain, subdomain_id=1)
# Intégration sur la surface extérieure marquée
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_exterior)

dx_int = ufl.Measure("dx", domain=interface_mesh)
Gamma_int = fem.assemble_scalar(fem.form(1 * dS(INT_TAG)))
Gamma_bulk = fem.assemble_scalar(fem.form(1 * dS(INNER_FACET_TAG)))
Gamma = fem.assemble_scalar(fem.form(1 * dx_int))
print(f"Total interface area = {Gamma_int:.3f}")
print(f"Total inner facets area = {Gamma_bulk:.3f}")
print(f"Total interface measure = {Gamma:.3f} vs {Gamma_int+Gamma_bulk:.3f}")

# Traction separation variables at the interface
V_int = fem.functionspace(interface_mesh, ("DG", 1))
d = fem.Function(V_int, name="Interfacial_damage")
d_prev = fem.Function(V_int, name="Previous_interfacial_damage")
d_old = fem.Function(V_int, name="Interfacial_damage_old")

# Traction separation law's functions


# ==========================================
# CZM: Propriétés Normales (Mode I)
# ==========================================
Gc_n = create_piecewise_constant_field(
    interface_mesh, facet_tags_domain, {INT_TAG: 6635}, name="Gc_n", default_value=1e6
)
sig_max_n = create_piecewise_constant_field(
    interface_mesh, facet_tags_domain, {INT_TAG: 1.5e6}, name="sigmax_n", default_value=3e6
)
delta_0_n = Gc_n / sig_max_n / ufl.exp(1)

# ==========================================
# CZM: Propriétés Tangentielles (Mode II)
# ==========================================
Gc_t = create_piecewise_constant_field(
    interface_mesh, facet_tags_domain, {INT_TAG: 15000}, name="Gc_t", default_value=2e6
)
sig_max_t = create_piecewise_constant_field(
    interface_mesh, facet_tags_domain, {INT_TAG: 3.0e6}, name="sigmax_t", default_value=4e6
)
delta_0_t = Gc_t / sig_max_t / ufl.exp(1)

beta = fem.Constant(domain, 2.0)


def normal_opening(v, n):
    return ufl.dot(v, n)


def tangential_opening(v, n):
    return v - normal_opening(v, n) * n

def normalized_effective_opening(v, n):
    dn = normal_opening(v, n)
    dt_vec = tangential_opening(v, n)
    
    # Macaulay bracket : Seule l'ouverture en traction (positive) génère du dommage
    dn_plus = ufl.max_value(dn, 0.0) 
    dt_sq = ufl.inner(dt_vec, dt_vec)
    
    # Lambda effectif adimensionnel (normalisé par dn0 et dt0)
    return ufl.sqrt((dn_plus / ufl.avg(delta_0_n))**2 + dt_sq / (ufl.avg(delta_0_t))**2)

def T(jump_v,jump_u, d, n):
    # 1. Variables basées sur la TrialFunction (du) -> POUR MULTIPLIER (Linéaire)
    dn = normal_opening(jump_v, n)
    dt_vec = tangential_opening(jump_v, n)
    # 2. Variables basées sur la Function connue (u) -> POUR TESTER (Non-linéaire)
    dn_u = normal_opening(jump_u, n)
    
    # Raideurs initiales indépendantes
    Kn_0 = ufl.avg(Gc_n / delta_0_n**2)
    Kt_0 = ufl.avg(Gc_t / delta_0_t**2)
    
    #heaviside = 0.5 * (ufl.sign(dn) + 1.0)    
    # Si compression (heaviside=0) -> Kn = Kn_0
    # Si traction (heaviside=1) -> Kn = Kn_0 * (1 - d)
    Kn = ufl.conditional(ufl.ge(dn_u, 0.0), Kn_0 * (1.0 - d), Kn_0)
    Kt = Kt_0 * (1 - d)
    
    # Vecteur traction reconstruit
    return Kn * dn * n + Kt * dt_vec


# Weak form formulation :
n_facet = ufl.FacetNormal(domain)
a_bulk = ufl.inner(sigma(du), epsilon(v)) * dx
a_interface = ufl.dot(T(ufl.jump(du),ufl.jump(u), ufl.avg(d),n_facet("-")), ufl.jump(v)) * dS(INT_TAG)+ufl.dot(T(ufl.jump(du),ufl.jump(u), ufl.avg(d),n_facet("-")), ufl.jump(v)) * dS(INNER_FACET_TAG)
a = a_bulk + a_interface


T_load = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
f = fem.Constant(domain, (0.0, 0,0.0))
Fext = ufl.dot(f, v) * dx + ufl.dot(T_load, v) * ds(LOAD_TAG)


facet_imap = domain.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts

domain_to_interfaces = np.full(num_facets, 1, dtype=np.int32) 
domain_to_interfaces[interfaces_cell_map] = np.arange(len(interfaces_cell_map), dtype=np.int32) #Mark la position des interfaces dans le domain global

entity_maps = {interface_mesh: domain_to_interfaces}
a_compiled = fem.form(a, entity_maps=entity_maps)
Fext_compiled = fem.form(Fext, entity_maps=entity_maps)

mesh_preview(domain,fdim,facets_interface)


#Interface quadrature intégration
n = ufl.FacetNormal(domain)
# Interface quadrature intégration
lambda_eff = normalized_effective_opening(ufl.jump(u), n_facet("-"))
# La variable adimensionnelle s'injecte directement dans l'exponentielle
d_expr = ufl.max_value(ufl.avg(d_prev), 1 - ufl.exp(-lambda_eff))

q_p = V_int.element.interpolation_points()
weights = np.full(q_p.shape[0], 1.0, dtype=dolfinx.default_scalar_type)
q_el = basix.ufl.quadrature_element(
    interface_mesh.basix_cell(), scheme="custom", points=q_p, weights=weights
)
Q = fem.functionspace(interface_mesh, q_el)
q_ = ufl.TestFunction(Q)
dS_custom = ufl.Measure(
    "dS",
    domain=domain,
    metadata={
        "quadrature_scheme": "custom",
        "quadrature_points": q_p,
        "quadrature_weights": weights,
    },
    subdomain_data=facet_tags_domain,
)

facet_interp = fem.form(
    1 / ufl.FacetArea(domain) * d_expr * ufl.avg(q_) * dS_custom,
    entity_maps=entity_maps,
)


#Boundary conditions
Uimp = fem.Constant(domain, (0, 0.0,0.0))

#Boundary conditions can be either do locate geometrically or topologicaly, pour les fonctions DG les cdl sont associées aux cell on ne peut pas définir les cdl topologiqument

left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.logical_and.reduce([np.isclose(x[0], 0.0),np.isclose(x[1],0.0)]))
right_dofs = fem.locate_dofs_geometrical(V, lambda x: np.logical_and.reduce([np.isclose(x[0], L),np.isclose(x[1],0.0)]))
load_dofs = fem.locate_dofs_geometrical(V, lambda x: np.logical_and.reduce([np.isclose(x[1], H),np.isclose(x[0],L/2)]))
bcs = [
    fem.dirichletbc(np.zeros((tdim,)), left_dofs, V),
    fem.dirichletbc(np.zeros((tdim,)), right_dofs, V),]
    #fem.dirichletbc(Uimp, load_dofs, V),]


v_reac = fem.Function(V)
fem.set_bc(v_reac.x.array, bcs)
virtual_work_form = fem.form(
    ufl.action(ufl.action(a, u) - Fext, v_reac), entity_maps=entity_maps
)

#Problem
problem = LinearProblem(
    a_compiled,
    Fext_compiled,
    u=u,
    bcs=bcs,
    petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
)

interface_damage = fem.form(ufl.avg(d) * dS(INT_TAG), entity_maps=entity_maps)
bulk_damage = fem.form(
    ufl.avg(d) * dS(INNER_FACET_TAG),
    entity_maps=entity_maps,
)


#Loop résolution
Nincr = 10
loading = np.linspace(0, -500, Nincr + 1)

Niter_max = 200
tol = 1e-6
damage_results = [[0.0, 0.0,0.0]]
Force = [0.0]
iterations = [0]
out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "a")


for i, t in enumerate(loading[1:]):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load step {i+1}/{len(loading)-1}")
    #Uimp.value[1] = t #Condition en déplacement imposé 
    T_load.value[1] = t #Condition en force
    nRes = 1.0
    j = 0
    while j < Niter_max:
        # displacement problem resolution
        problem.solve()
        # interpolation of damage on facets
        d.x.array[:] = fem.assemble_vector(facet_interp).array
        # normalized residual for convergence check
        nRes = (
            np.sqrt((fem.assemble_scalar(fem.form((d - d_old) ** 2 * dx_int)))) / Gamma
        )
        d_old.x.array[:] = d.x.array[:]
        j += 1
        print(f" Iteration {j:2d} | Residual: {nRes}")
        if nRes < tol:
            break
    else:
        raise ValueError(
            "Fixed-point solver did not converge in less than {} iterations".format(
                Niter_max
            )
        )
    iterations.append(j)
    d_prev.x.array[:] = d.x.array[:]
    Force.append(fem.assemble_scalar(virtual_work_form))
    damage_results.append(
        [fem.assemble_scalar(interface_damage), fem.assemble_scalar(bulk_damage)]
    )
    print("damage max", max(d.x.array[:]))
    out_file.write_function(u, i)
    #out_file.write_function(d, i)
    clear_output(wait=True)

out_file.close()


# Export the parent domain tags to ParaView
with io.XDMFFile(domain.comm, "results/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    # This writes your 0 and 1 values as a color map
    xdmf.write_meshtags(facet_tags_domain, domain.geometry)