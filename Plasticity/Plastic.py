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
import matplotlib.pyplot as plt
# ==========================================
# 1. Paramètres du modèle
# ==========================================
L, H, e = 5.0, 1.0, 0.4  # Dimensions de la poutre (m)
x_interface = H / 2.0  # Position de l'interface

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
    return np.isclose(x[1], x_interface, atol=1e-6) #np.logical_and.reduce([np.isclose(x[1], x_interface), x[0] >= 0.0, x[0] <= L/2.0])

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
#Ealsticity tensor
u = fem.Function(V, name="Displacement")
#Plasticity tensor
u_p=fem.Function(V, name="Plastic_displacement")

#Plasticity variables
W_tensor = fem.functionspace(domain, ("DG", 0, (tdim, tdim)))
sigma_eval = fem.Function(W_tensor)
sigma_expr = fem.Expression(sigma(u), W_tensor.element.interpolation_points())
W_scalar = fem.functionspace(domain, ("DG", 0))
eps_p = fem.Function(W_tensor, name="Plastic_Strain")
kappa_accum = fem.Function(W_scalar, name="Accumulated_Plasticity")

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


# Integration in domain
dx = ufl.Measure("dx", domain=domain)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags_domain, subdomain_id=1)
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



Gc = create_piecewise_constant_field(
    interface_mesh, facet_tags_domain, {INT_TAG: 6635}, name="Gc", default_value=1e6#12235
)
sig_max = create_piecewise_constant_field(
    interface_mesh,
    facet_tags_domain,
    {INT_TAG: 1.5e6 }, #1.5e6
    name="sigmax",
    default_value=3e6,
)

delta_0 = Gc / sig_max / ufl.exp(1)
beta = fem.Constant(domain, 2.0)


def normal_opening(v, n):
    return ufl.dot(v, n)


def tangential_opening(v, n):
    return v - normal_opening(v, n) * n

def effective_opening(v, n):
    return ufl.sqrt(normal_opening(v, n) ** 2 + beta**2 * tangential_opening(v, n) ** 2)

def T(opening, d):
    K_int = ufl.avg(Gc / delta_0**2) * (1 - d)
    return K_int * opening

#Failure envelope:


def check_mohr_coulomb_yield(stress_tensors, phi, c):
    """
    Evaluates the Mohr-Coulomb yield criterion for an array of stress tensors.
    stress_tensors shape: (N_cells, 3, 3)
    """
    # 1. Compute principal stresses (eigenvalues) for all cells at once
    # np.linalg.eigvalsh returns them in ascending order: sigma3, sigma2, sigma1
    eigvals = np.linalg.eigvalsh(stress_tensors)
    
    sigma3 = eigvals[:, 0] # Most compressive (or least tensile)
    sigma1 = eigvals[:, 2] # Most tensile
    
    # 2. Compute trigonometric constants
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # 3. Evaluate the yield function 'f' for all cells
    # f = sigma1 - sigma3 * N(phi) - 2 * c * sqrt(N(phi))
    term1 = (1 + sin_phi) / (1 - sin_phi)   
    term2 = (2 * c * cos_phi) / (1 - sin_phi)
    f = sigma1 - sigma3 * term1 - term2
    
    # 4. Return a boolean mask of yielding cells and the f-values
    is_yielding = f > 0
    return is_yielding, f



# Weak form formulation :
a_bulk = ufl.inner(sigma(du), epsilon(v)) * dx
a_interface = (ufl.dot(T(ufl.jump(du), ufl.avg(d)), ufl.jump(v)) * dS(INT_TAG) + 
               ufl.dot(T(ufl.jump(du), ufl.avg(d)), ufl.jump(v)) * dS(INNER_FACET_TAG))

a = a_bulk + a_interface

f = fem.Constant(domain, (0.0, 0.0, 0.0))

# --- CORRECTION ---
# Function to calculate stress directly from a strain tensor
def sigma_plastic(eps_tensor):
    return lmbda * ufl.tr(eps_tensor) * ufl.Identity(tdim) + 2.0 * mu * eps_tensor

# L_bulk now correctly evaluates the plastic stress from the return mapping tensor
L_bulk = ufl.inner(sigma_plastic(eps_p), epsilon(v)) * dx

# Note: Since your return mapping only computes bulk plasticity (eps_p), 
# there is no interface plastic jump calculated. Thus, L_interface for plasticity 
# should be 0 unless you implement an interface friction return mapping.
L_interface = 0 

Fext = ufl.dot(f, v) * dx 
RHS = Fext + L_bulk + L_interface


facet_imap = domain.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts

domain_to_interfaces = np.full(num_facets, 1, dtype=np.int32) 
domain_to_interfaces[interfaces_cell_map] = np.arange(len(interfaces_cell_map), dtype=np.int32) #Mark la position des interfaces dans le domain global

entity_maps = {interface_mesh: domain_to_interfaces}
a_compiled = fem.form(a, entity_maps=entity_maps)
RHS_compiled = fem.form(RHS, entity_maps=entity_maps)

#mesh_preview(domain,fdim,facets_interface)


#Interface quadrature intégration
n = ufl.FacetNormal(domain)
delta = effective_opening(ufl.jump(u), n("-"))
d_expr = ufl.max_value(ufl.avg(d_prev), 1 - ufl.exp(-delta / ufl.avg(delta_0)))

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

left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
right_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], L))
load_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], H))
bcs = [
    fem.dirichletbc(np.zeros((tdim,)), left_dofs, V),
    #fem.dirichletbc(np.zeros((tdim,)), right_dofs, V),
    fem.dirichletbc(Uimp, load_dofs, V),]


v_reac = fem.Function(V)
fem.set_bc(v_reac.x.array, bcs)
virtual_work_form = fem.form(
    ufl.action(ufl.action(a, u) - RHS, v_reac), entity_maps=entity_maps
)

#Problem
problem = LinearProblem(
    a_compiled,
    RHS_compiled,
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
loading = np.concatenate([np.linspace(0, -0.0000000005, Nincr + 1),np.linspace( -0.0000000005,0, Nincr + 1)])

Niter_max = 200
tol = 1e-6
damage_results = [[0.0, 0.0,0.0]]
Force = [0.0]
iterations = [0]
out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "a")


for i, t in enumerate(loading[1:]):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load step {i+1}/{len(loading)-1}")
    Uimp.value[1] = t
    nRes = 1.0
    j = 0
    while j < Niter_max:
        # displacement problem resolution
        problem.solve()
        #Plasticity criterion + return mapping
        # 2. Extract numerical stresses from the current displacement 'u'
        sigma_eval.interpolate(sigma_expr)
    
        # Reshape the flat array into a list of 3x3 matrices: shape (N_cells, 3, 3)
        stress_arrays = sigma_eval.x.array.reshape(-1, 3, 3)
    
        #==========================================================
        # 3. THE MOHR-COULOMB CRITERION CHECK
        # ==========================================================
        phi=30
        psi=phi #Dilatancy angle #assuming associated flow rule for simplicity

        def N(angle):
            return (1 + np.sin(angle)) / (1 - np.sin(angle))
        
        

        # Fictitious softening parameters for concrete cohesion
        c_0 = 0.1      # Initial cohesion
        c_res = c_0 * 0.1 # Residual cohesion (it won't drop below this)
        eta = 50.0    # Softening rate (how fast it drops)

        is_yielding, f_values = check_mohr_coulomb_yield(stress_arrays, phi, c_res)
    
        # --- Constants defined outside the cell loop ---
        if np.any(is_yielding):
            yielding_cells = np.where(is_yielding)[0]
            print("Yielding")
            for cell in yielding_cells:
                # 1. Get trial stress and eigenvectors
                sig_trial_3x3 = stress_arrays[cell]
                vals, vecs = np.linalg.eigh(sig_trial_3x3)
                
                # Local elastic properties (replace with your E_func extraction)
                # Local elastic properties dynamically extracted from E_func
                E_val = E_func.x.array[cell]
                nu_val = 0.2  # As defined in your global parameters
                
                # Calculate Lamé parameters
                mu_val = E_val / (2.0 * (1.0 + nu_val))
                lam_val = (E_val * nu_val) / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

                # The constant elastic part of the derivative (A)
                A = 2 * mu_val * (1 + N(phi) * N(psi)) + lam_val * (1 - N(psi)) * (1 - N(phi))
            
                # 2. Calculer la valeur exacte de f_trial
                f_trial = vals[2] - vals[0] * N(phi) - 2 * c_0 * np.sqrt(N(phi))
                
                # 3. ESTIMATION INITIALE OPTIMALE : La solution analytique parfaite
                delta_gamma = f_trial / A

                # Retrieve the accumulated plastic history for this specific cell
                kappa_old = kappa_accum.x.array[cell]                
                # --- LOCAL NEWTON-RAPHSON LOOP ---                    
                # 1. Build the principal plastic strain increment tensor for this iteration
                d_eps_prin = np.diag([
                    -N(psi) * delta_gamma, # sig3 direction
                    0.0,                  # sig2 direction
                    1.0 * delta_gamma     # sig1 direction
                ])
                
                # 2. Calculate Delta Kappa directly using the double dot product (eps_p : eps_p)
                # np.sum(A * A) perfectly calculates the double dot product in NumPy
                delta_kappa = np.sqrt((2.0 / 3.0) * np.sum(d_eps_prin * d_eps_prin))
                
                # A. Update internal state based on current guess
                kappa_current = kappa_old 
                
                # B. Evaluate the softening law (Exponential decay of cohesion)
                c_current = c_res + (c_0 - c_res) * np.exp(-eta * kappa_current)
                
                # C. Evaluate the derivative (dc / d(delta_gamma))
                # BE CAREFUL: You STILL need the analytical derivative of delta_kappa with 
                # respect to delta_gamma for the chain rule, otherwise Newton-Raphson will fail!
                dkappa_dgamma = np.sqrt((2.0 / 3.0) * (1.0 + N(psi)**2)) 
                dc_dgamma = -eta * (c_0 - c_res) * np.exp(-eta * kappa_current) * dkappa_dgamma
                
                max_local_iters = 25
                tol_local = 1e-6
                
                for k in range(max_local_iters):
                    # A. Update internal state based on current guess
                    kappa_current = kappa_old + (delta_gamma * dkappa_dgamma)
                    
                    # B. Evaluate the softening law (Exponential decay of cohesion)
                    c_current = c_res + (c_0 - c_res) * np.exp(-eta * kappa_current)
                    # C. Evaluate the derivative of the softening law (dc / d(delta_gamma)) with the chain rule
                    dc_dgamma = -eta * (c_0 - c_res) * np.exp(-eta * kappa_current) * dkappa_dgamma
                    
                    # D. Calculate the current principal stresses pulled back by delta_gamma
                    # sig_current = sig_trial - delta_gamma * C : dg/dsigma
                    sig1_current = vals[2] - delta_gamma * (2 * mu_val + lam_val * (1 - N(psi)))
                    sig3_current = vals[0] - delta_gamma * (-2 * mu_val * N(psi) + lam_val * (1 - N(psi)))
                    
                    # E. Calculate the Residual (Current yield function value)
                    R = sig1_current - sig3_current * N(phi) - 2 * c_current * np.sqrt(N(phi))
                    
                    # F. Check convergence
                    if abs(R) < tol_local:
                        break
                        
                    # G. Calculate the Derivative of the Residual (dR/d(delta_gamma))
                    dR_dgamma = -A - 2 * dc_dgamma * np.sqrt(N(phi))
                    
                    # H. Update the guess
                    delta_gamma = delta_gamma - (R / dR_dgamma)
                    
                else:
                    # If the loop finishes without breaking, it failed to converge
                    print(f"Warning: Local NR failed at cell {cell}, Residual: {R:.3e}")
                
                # --- END OF NEWTON-RAPHSON ---
                # 4. We now have the correct delta_gamma. Define principal plastic strains.
                d_eps_prin = np.diag([
                    -N(psi) * delta_gamma, # sig3 direction
                    0.0,                  # sig2 direction
                    1.0 * delta_gamma     # sig1 direction
                ])
                
                # 5. Rotate back to Global Coordinates
                delta_eps_global = vecs @ d_eps_prin @ vecs.T
                
                # 6. Update FEniCSx state variables
                # Update plastic strain tensor
                current_eps_p = eps_p.x.array[cell * 9 : (cell + 1) * 9].reshape(3, 3)
                updated_eps_p = current_eps_p + delta_eps_global
                eps_p.x.array[cell * 9 : (cell + 1) * 9] = updated_eps_p.flatten()
                
                # Update the accumulated plastic history
                kappa_accum.x.array[cell] = kappa_current

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

    out_file.write_function(u, i)
    #out_file.write_function(d, i)
    clear_output(wait=True)

out_file.close()




# Export the parent domain tags to ParaView
with io.XDMFFile(domain.comm, "results/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    # This writes your 0 and 1 values as a color map
    xdmf.write_meshtags(facet_tags_domain, domain.geometry)
