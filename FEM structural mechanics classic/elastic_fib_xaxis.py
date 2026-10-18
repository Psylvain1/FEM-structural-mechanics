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
from geometry_fibV2_xaxis import *
import vtk
import matplotlib.pyplot as plt
 



# Propriétés Béton 1 du coffrage
E1 = 33e9  # 36 GPa
nu1 = 0.2

# Propriétés Béton 2 de remplissage
E2 = 31e9  # 38 GPa
nu2 = 0.2

#Propriétés de l'acier
Eacier=210e9
nu3 = 0.3

largeur, H, e_3DP, L,es, phi=create_geometry()
domain, cell_tags, facet_tags = io.gmshio.read_from_msh("poutre.msh", MPI.COMM_WORLD, 0, gdim=3)


# ---------------------------------------------------------
# CORRECTION FENICSX 0.9.0 : functionspace unifié
# On précise (domain.geometry.dim,) pour en faire un vecteur
# ---------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# Espaces pour définir les propriétés hétérogènes (CG0)
V_E = fem.functionspace(domain, ("DG", 0))  # FEniCSx 0.9.0
E_func = fem.Function(V_E)
nu_func = fem.Function(V_E)

# On utilise la méthode .find(tag_id) sur l'objet cell_tags
cells_U = cell_tags.find(1)      # Tag 1 = Coffrage_U
cells_Noyau = cell_tags.find(2)  # Tag 2 = Coeur_Coule
cells_Aciers = cell_tags.find(3) # Tag 3 = Aciers

# ==========================================
# 4. Assignation des propriétés matérielles
# ==========================================
# On assigne les modules d'Young (en Pa) directement aux indices des cellules trouvées
E_func.x.array[cells_U] = E1          # Béton 1 (ex: 36e9)
E_func.x.array[cells_Noyau] = E2      # Béton 2 (ex: 38e9)
E_func.x.array[cells_Aciers] = Eacier  # Acier (ex: 210 GPa)

nu_func.x.array[cells_U] = nu1         # Béton 1 (ex: 36e9)
nu_func.x.array[cells_Noyau] = nu2     # Béton 2 (ex: 38e9)
nu_func.x.array[cells_Aciers] = nu3  # Acier (ex: 210 GPa)



mu = E_func / (2.0 * (1.0 + nu_func))
lmbda = E_func * nu_func / ((1.0 + nu_func) * (1.0 - 2.0 * nu_func))

def epsilon(v):
    return ufl.sym(ufl.grad(v))

def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(tdim) + 2.0 * mu * epsilon(v)

# Weak form with a discontinuous displacement.
tdim = domain.topology.dim
V = fem.functionspace(domain, ("CG", 1, (tdim,)))
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# Création d'un meshtag pour restreindre l'intégrale dS à l'interface
# (Nécessaire dans les versions récentes pour cibler précisément la discontinuité)
fdim = domain.topology.dim - 1

#Load Tags :

# --- NOUVEAU : Marqueur pour la charge surfacique ---
LOAD_TAG = 3

# Par exemple, on charge une zone de 20 cm au centre, sur la face supérieure (H)
def load_surface_locator(x):
    return  np.logical_and.reduce([np.isclose(x[1], H, atol=1e-5),np.isclose(x[0], L/2.0, atol=1e-1)]) #np.isclose(x[1], H, atol=1e-5) 


# Trouver les facettes frontières (extérieures) correspondant à cette zone
facets_load = mesh.locate_entities_boundary(domain, fdim, load_surface_locator)
load_values = np.full_like(facets_load, LOAD_TAG, dtype=np.int32)

# Créer un tag spécifique pour les bords extérieurs
facet_tags_exterior = mesh.meshtags(domain, fdim, facets_load, load_values)


# Integration in domain
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain,subdomain_data=facet_tags_exterior)


# Weak form formulation :
a_bulk = ufl.inner(sigma(du), epsilon(v)) * dx
a = a_bulk 



# 2. Constante d'intensité du couple
Force_surf = fem.Constant(domain, default_scalar_type(0.0)) 

# 3. Champ de traction tangentiel (Rotation autour de Z)
T_load = ufl.as_vector((
    0,  # Force X proportionnelle à la distance en Y
    Force_surf/(largeur*2e-1),  # Force Y proportionnelle à la distance en X
     0.0                # Aucune force axiale (sur Z)
))

# Propriétés de masse volumique (en kg/m^3)
rho1 = 2100.0  # Béton 1
rho2 = 2100.0  # Béton 2
rho3 = 7850.0  # Acier
g_val = 9.81   # Accélération de la pesanteur (m/s^2)

# Création de la fonction pour la masse volumique
rho_func = fem.Function(V_E)
rho_func.x.array[cells_U] = rho1
rho_func.x.array[cells_Noyau] = rho2
rho_func.x.array[cells_Aciers] = rho3

f = ufl.as_vector((
    0.0, 
    -rho_func * g_val,  # Gravité vers le bas
    0.0
))

Fext = ufl.dot(f, v) * dx + ufl.dot(T_load, v) * ds(LOAD_TAG)

facet_imap = domain.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts

a_compiled = fem.form(a)
Fext_compiled = fem.form(Fext)


camera =[
        (L/2, 0.3,  4.0),  # Position de votre œil
        (L/2, -0.1, 0.1),  # Le point que vous regardiez
        (0.0, 1.0, 0.0)   # L'axe qui pointe vers le haut
    ]   
mesh_preview(domain, cell_tags,camera)

#Boundary conditions can be either do locate geometrically or topologicaly, pour les fonctions CG les cdl sont associées aux cell on ne peut pas définir les cdl topologiqument
d_appui = 0.25

# --- CORRECTION POUR LES SOUS-ESPACES ---
# On effondre (collapse) les sous-espaces pour lire leur géométrie
V0, _ = V.sub(0).collapse()
V1, _ = V.sub(1).collapse()

eps=1e-2

# 1. Appui droit : on encastre une ZONE SURFACIQUE et non plus une ligne
left_dofs = fem.locate_dofs_geometrical(
    V, 
    lambda x: np.logical_and.reduce([
        np.isclose(x[1], 0.0, atol=eps),np.isclose(x[0], d_appui, atol=eps)
    ])
)

# --- Effondrement des sous-espaces pour les appuis à rouleau ---
V0, _ = V.sub(0).collapse()
V1, _ = V.sub(1).collapse()

# 2. Appui gauche : Maintien en X sur la surface
right_dofs = fem.locate_dofs_geometrical(
    (V.sub(0), V0), 
    lambda x: np.logical_and.reduce([
        np.isclose(x[1], 0.0, atol=eps),
        np.isclose(x[0], L-d_appui, atol=eps)
    ])
)

# 3. Appui gauche : Maintien en Y sur la même surface
right_dofsB = fem.locate_dofs_geometrical(
    (V.sub(1), V1), 
    lambda x: np.logical_and.reduce([
        np.isclose(x[1], 0.0, atol=eps),
        np.isclose(x[0], L-d_appui, atol=eps)   ])
)

bcs = [
    fem.dirichletbc(np.zeros((tdim,)), left_dofs, V),
    fem.dirichletbc(default_scalar_type(0.0), right_dofs[0], V.sub(0)),
    fem.dirichletbc(default_scalar_type(0.0), right_dofsB[0], V.sub(1)), 
]


#Problem
problem = LinearProblem(
    a_compiled,
    Fext_compiled,
    u=u,
    bcs=bcs,
    petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
    #petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
)


#Loop résolution
Nincr = 2
loading = np.linspace(-30000, -30000, Nincr + 1) #Peak Load 100 kN

Niter_max = 200
tol = 1e-6
damage_results = [[0.0, 0.0,0.0]]
Force = [0.0]
iterations = [0]
out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "a")

# Au lieu d'un espace tensoriel, on crée un espace SCALAIRE discontinu 
# pour accueillir chaque composante séparément.
V_scalar = fem.functionspace(domain, ("DG", 0))

# Dictionnaire des composantes qui nous intéressent (matrice symétrique)
# Rappel des indices : x=0, y=1, z=2
components = {
    "sigma_xx": (0, 0),
    "sigma_yy": (1, 1),
    "sigma_zz": (2, 2),
    "tau_xy": (0, 1), # Cisaillement dans le plan XY
    "tau_xz": (0, 2), # Cisaillement (Torsion) plan XZ
    "tau_yz": (1, 2)  # Cisaillement (Torsion) plan YZ
}

# Listes pour stocker nos Fonctions (pour ParaView) et nos Expressions (pour le calcul)
list_stress_funcs = []
list_stress_exprs = []

for name, (row, col) in components.items():
    # 1. On crée la fonction avec le beau nom pour ParaView
    func = fem.Function(V_scalar, name=name)
    
    # 2. On crée l'expression UFL qui cible UNIQUEMENT la case [row, col] du tenseur
    expr = fem.Expression(sigma(u)[row, col], V_scalar.element.interpolation_points())
    
    list_stress_funcs.append(func)
    list_stress_exprs.append(expr)

# On garde le même fichier de sortie (mode "w" conseillé pour réinitialiser le fichier au lancement)

stress_file = io.VTKFile(MPI.COMM_WORLD, "results/stress.pvd", "a")


# ==========================================
# PREPARATION POUR L'EXTRACTION DE TAU_ZY
# ==========================================
# Coordonnées du point : on décale très légèrement L pour être sûr de tomber dans le volume
point_eval = np.array([[e_3DP, H, L/2]], dtype=np.float64)

# Création de l'arbre de recherche pour trouver la cellule contenant le point
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point_eval)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, point_eval)

points_on_proc = []
cells_for_eval = []

# Vérifie si le point est sur le processeur actuel (utile en calcul parallèle MPI)
if len(colliding_cells.links(0)) > 0:
    points_on_proc.append(point_eval[0])
    cells_for_eval.append(colliding_cells.links(0)[0])

# Variables pour stocker l'historique
load_history = [0.0]  # On initialise avec le point de départ (charge nulle)
tau_zy_history = [0.0]

# Trouver l'indice de tau_yz dans votre liste (pour être robuste)
idx_tau_yz = list(components.keys()).index("sigma_zz")


# =====================================================================
# INITIALISATION DE L'ANIMATION PYVISTA (AVANT LA BOUCLE)
# =====================================================================
if MPI.COMM_WORLD.rank == 0:
    # 1. Extraire la topologie et les coordonnées initiales (non déformées)
    topology, cell_types, x_initial = plot.vtk_mesh(V)
    mesh_anim = pv.UnstructuredGrid(topology, cell_types, x_initial)
    
    # 2. Configurer le Plotter en mode hors-écran pour ne pas bloquer le calcul
    plotter_anim = pv.Plotter(notebook=False, off_screen=True,window_size=[1920, 1080])

    #plotter_anim.add_text("Déformation de la poutre en torsion", font_size=12)
    
    # 3. Initialiser le champ de couleur avec la norme du déplacement (0 au départ)
    #disp_initial = u.x.array.reshape((-1, domain.geometry.dim))
    #mesh_anim.point_data["Deplacement_Norme"] = np.linalg.norm(disp_initial, axis=1)

    # Initialiser le champ Tau (des zéros pour commencer)
    mesh_anim.cell_data[r"$\sigma_{zz}$"] = np.zeros(mesh_anim.n_cells)
    # 4. Ajouter la maille au plotter avec une échelle adaptée à la physique (très petite !)
    clim_max = 1e6 #1e-9  # Amplitude maximale attendue pour c = -50
    # Configuration de la barre de couleur
    
    # 1. On vide le titre de la barre de couleur
    sargs = {
        "title": "",  # <-- Plus de titre ici !
        "label_font_size": 32,
        "position_x": 0.2,  # On place la barre à droite
        "height": 0.2,       # On réduit un peu la hauteur pour laisser de la place en haut
        "width": 0.6
    }

    # 2. On ajoute le titre de manière 100% indépendante
    titre_actor=plotter_anim.add_text(
        r"$\sigma_{zz}$", 
        position=(0.5, 0.175), # 85% vers la droite, 90% vers le haut
        viewport=True,         # <-- Indispensable pour que PyVista comprenne les %
        font_size=32,
        name="tau_title"         # Le nommer permet de l'écraser si besoin dans la boucle
    )
    titre_actor.prop.justification_horizontal = 'center'

    actor_anim = plotter_anim.add_mesh(
        mesh_anim, 
        scalars=r"$\sigma_{zz}$", 
        cmap="jet", 
        show_edges=True, #True,
        clim=[-clim_max, clim_max], 
        opacity=1,
        scalar_bar_args=sargs  # <-- On injecte les paramètres ici
    )
    
    # juste après actor_anim = plotter_anim.add_mesh(...)
    #plotter_anim.view_isometric()  # Vue 3D standard (Isométrique)
    # ---------------------------------------------------------
    # ON INJECTE VOTRE ANGLE PERSONNALISÉ ICI :
    # ---------------------------------------------------------
    plotter_anim.camera_position = camera
        

    # 5. Ouvrir le fichier de destination GIF
    plotter_anim.open_gif("results/TractionHP_poutre.gif")
    plotter_anim.show_axes()

for i, t in enumerate(loading[1:]):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load step {i+1}/{len(loading)-1}")
    #Uimp.value[1] = t #Condition en déplacement imposé 
    #T_load.value[1] = t #Condition en force
    # CORRECT : On modifie la constante scalaire c
    Force_surf.value = np.array(t, dtype=default_scalar_type)
    problem.solve()
    out_file.write_function(u, i)

    # 1. On interpole l'expression de chaque composante dans sa fonction respective
    for func, expr in zip(list_stress_funcs, list_stress_exprs):
        func.interpolate(expr)
    # 2. On écrit TOUTE la liste d'un coup dans le fichier VTK/PVD
    stress_file.write_function(list_stress_funcs, i)
    # ---------------------------------------------------------
    # EXTRACTION DE LA VALEUR TAU_ZY
    # ---------------------------------------------------------
    tau_func = list_stress_funcs[idx_tau_yz]
    
    if len(points_on_proc) > 0:
        # Évaluation locale
        val_locale = tau_func.eval(points_on_proc, cells_for_eval)[0]
    else:
        val_locale = 0.0
        
    # Synchronisation si exécution en parallèle (MPI)
    val_globale = MPI.COMM_WORLD.allreduce(val_locale, op=MPI.SUM)
    
    if MPI.COMM_WORLD.rank == 0:
        load_history.append(t)
        tau_zy_history.append(val_globale)
    # -------------------------

    # CAPTURE DE L'IMAGE POUR L'ANIMATION PYVISTA
    # -----------------------------------------------------------------
    if MPI.COMM_WORLD.rank == 0:
        # Extraire le déplacement actuel et calculer la norme de chaque nœud
        disp_array = u.x.array.reshape((-1, domain.geometry.dim))
        norme_disp = np.linalg.norm(disp_array, axis=1)
        
        amplification = 5.0
        # Affichage de contrôle dans ton terminal
        #print(f"   -> Déplacement max réel : {max_real_disp:.3e} m | Amplification : {amplification:.1e}")
        
        # CORRECTION 1 : Assignation directe (sans [:]) pour forcer VTK à mettre à jour la forme
        mesh_anim.points[:] = x_initial + disp_array * amplification
        
        # Mettre à jour le champ de couleur
        #mesh_anim.point_data["Deplacement_Norme"] = norme_disp
        
        tau_array = tau_func.x.array[:mesh_anim.n_cells]
        
        # On l'assigne à cell_data !
        mesh_anim.cell_data[r"$\sigma_{zz}$"] =tau_array
        
        # OPTIONNEL : Ajuster dynamiquement l'échelle de couleur de la légende
        #plotter_anim.update_scalar_bar_range([0, max_real_disp if max_real_disp > 1e-15 else 1.0])
        # On récupère l'objet "barre de couleur" actif
        
        plotter_anim.add_text(
            f"Pas {i+1}/{len(loading)-1} | force = {t*1e-3:.0f} kN", 
            font_size=12, 
            name="status_text"
        )        
        
        # CORRECTION 2 : Forcer le rendu de la nouvelle géométrie
        plotter_anim.render()
        plotter_anim.write_frame()

    clear_output(wait=True)

out_file.close()
stress_file.close()

# =====================================================================
# CRITÈRE GLOBAL POUR L'ÉTUDE DE CONVERGENCE DE MAILLAGE
# =====================================================================
# Formulation de l'énergie de déformation élastique : U = 1/2 * int(sigma : epsilon) dx
energy_form = 0.5 * ufl.inner(sigma(u), epsilon(u)) * dx
energy_compiled = fem.form(energy_form)

# Calcul de l'intégrale (localement sur le processeur)
energy_local = fem.assemble_scalar(energy_compiled)

# Synchronisation MPI pour obtenir la valeur globale (si exécution en parallèle)
energy_global = MPI.COMM_WORLD.allreduce(energy_local, op=MPI.SUM)

# Récupération du nombre total de degrés de liberté (DOFs)
# index_map.size_global donne le nombre de nœuds, index_map_bs donne la dimension (3 pour 3D)
num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

if MPI.COMM_WORLD.rank == 0:
    print("\n" + "="*50)
    print(" RÉSULTAT POUR L'ÉTUDE DE CONVERGENCE DE MAILLAGE")
    print("="*50)
    print(f"Fichier maillage utilisé : poutre.msh")
    print(f"Nombre total de DDL (DOFs) : {num_dofs_global}")
    print(f"Énergie de déformation totale : {energy_global:.6e} Joules")
    print("="*50 + "\n")

# Extraction de la composante Y du déplacement
u_y = u.sub(1).collapse()

# Trouver la valeur absolue maximale du déplacement vertical sur le processeur local
min_uy_local = u_y.x.array.min()  # Négatif car la poutre fléchit vers le bas
max_deflection_local = abs(min_uy_local)

# Synchronisation MPI pour trouver le max sur tous les processeurs
max_deflection_global = MPI.COMM_WORLD.allreduce(max_deflection_local, op=MPI.MAX)

if MPI.COMM_WORLD.rank == 0:
    print(f"Flèche maximale (déplacement Y) : {max_deflection_global * 1000:.4f} mm")


# ---------------------------------------------------------
# NOUVEAU : Fonction pour le code couleur ParaView
# ---------------------------------------------------------
mat_func = fem.Function(V_E, name="Material_ID")
mat_func.x.array[cells_U] = 1.0       # Couleur 1
mat_func.x.array[cells_Noyau] = 2.0   # Couleur 2
mat_func.x.array[cells_Aciers] = 3.0  # Couleur 3

# Exportation immédiate (hors de la boucle temporelle)
with io.VTKFile(MPI.COMM_WORLD, "results/materials.pvd", "w") as mat_file:
    mat_file.write_function(mat_func, 0.0)
# ---------------------------------------------------------

# Fermer le plotter d'animation une fois la boucle terminée
if MPI.COMM_WORLD.rank == 0:
    # --- NOUVEAU : Ajouter des frames figées à la fin ---
    frames_to_hold = 20  # Nombre de frames pour la pause (ex: 20 frames = ~2 secondes à 10 fps)
    for _ in range(frames_to_hold):
        plotter_anim.write_frame()

    plotter_anim.close()
    
    # --- AJOUT : Détruit explicitement l'objet pour éviter l'erreur VTK à la fermeture ---
    del plotter_anim 
    
    print("L'animation 3D a été enregistrée avec succès dans 'results/TractionHP_poutre.gif'")




 

# ==========================================
# TRACÉ DE LA COURBE TAU_ZY = f(Couple)
# ==========================================
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    graphics_setup() 
    # On prend la valeur absolue de la charge si elle est négative (convention)
    loads_plot = np.abs(load_history)*1e-3  # Conversion en kN pour l'axe X
    tau_zy_history = np.abs(np.array(tau_zy_history)*1e-6)  # Conversion en tableau NumPy pour un tracé plus facile    

    plt.plot(loads_plot, tau_zy_history, marker='o', linestyle='-', color='#d62728', linewidth=2)
    plt.xlabel("Force en [kN]", fontsize=12)
    plt.ylabel(r"$\sigma_{zz}$ [MPa]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0,1)
    # Sauvegarde et affichage
    plt.tight_layout()
    plt.savefig("results/courbe_sigma_xx.pdf", bbox_inches='tight')
    #plt.show()

