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
from RCcolumn import *
import vtk

 


# Propriétés Béton 1 du coffrage
E1 = 30e9  # 36 GPa
nu1 = 0.2

# Propriétés Béton 2 de remplissage
E2 = 30e9  # 38 GPa
nu2 = 0.2



largeur, H, e_3DP, L,es, phi=create_geometry()

domain, cell_tags, facet_tags = io.gmshio.read_from_msh("colonne.msh", MPI.COMM_WORLD, 0, gdim=3)




# Espaces pour définir les propriétés hétérogènes (CG0)
V_E = fem.functionspace(domain, ("DG", 0))  # FEniCSx 0.9.0
E_func = fem.Function(V_E)

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
E_func.x.array[cells_Aciers] = 210e9  # Acier (ex: 210 GPa)

nu = 0.2
mu = E_func / (2.0 * (1.0 + nu))
lmbda = E_func * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

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
    return np.isclose(x[2], L,atol=1e-5) 
"""
def load_surface_locator(x):
    # Parameters for the 0.34 section and 0.2 rectangle
    x_center = 0.17  # L / 2.0
    y_center = 0.17  # H / 2.0
    half_size = 0.10 # 0.2 / 2.0
    
    # Store all geometric constraints in a list
    conditions = [
        np.isclose(x[2], L, atol=1e-5),  # On the Z plane
        x[0] >= (x_center - half_size),            # X min boundary (0.07)
        x[0] <= (x_center + half_size),            # X max boundary (0.27)
        x[1] >= (y_center - half_size),            # Y min boundary (0.07)
        x[1] <= (y_center + half_size)             # Y max boundary (0.27)
    ]
    # Apply a logical AND across all conditions simultaneously
    return np.logical_and.reduce(conditions)
"""


# Trouver les facettes frontières (extérieures) correspondant à cette zone
facets_load = mesh.locate_entities_boundary(domain, fdim, load_surface_locator)
load_values = np.full_like(facets_load, LOAD_TAG, dtype=np.int32)

# Créer un tag spécifique pour les bords extérieurs
facet_tags_exterior = mesh.meshtags(domain, fdim, facets_load, load_values)


# Integration in domain
dx = ufl.Measure("dx", domain=domain)
# Intégration sur la surface extérieure marquée
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_exterior)


# Weak form formulation :
a_bulk = ufl.inner(sigma(du), epsilon(v)) * dx
a = a_bulk 


# =====================================================================
# MODIFICATION : Définition du centre de rotation de la section
# =====================================================================
# =====================================================================
# MODIFICATION : Torsion autour de l'axe Z
# =====================================================================
Y_c = H / 2.0        # Centre en Y
X_c = largeur / 2.0  # Centre en X

# 1. Récupérer les coordonnées spatiales
x = ufl.SpatialCoordinate(domain)

# 2. Constante d'intensité du couple
Force_surf = fem.Constant(domain, default_scalar_type(0.0)) 

# 3. Champ de traction tangentiel (Rotation autour de Z)
T_load = ufl.as_vector((
    0, 
    0,  
    Force_surf/(largeur*H)                
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
    0.0,
    -rho_func * g_val  # Gravité vers le bas
))
Fext = ufl.dot(f, v) * dx + ufl.dot(T_load, v) * ds(LOAD_TAG)


a_compiled = fem.form(a)
Fext_compiled = fem.form(Fext)


camera =[
        (2.5, 1, 1.7),  # Position de votre œil
        (0.1, 0.1, 0.35),  # Le point que vous regardiez
        (0.0, 0, 1.0)   # L'axe qui pointe vers le haut
    ]   
#mesh_preview(domain, cell_tags,camera)




#Boundary conditions can be either do locate geometrically or topologicaly, pour les fonctions CG les cdl sont associées aux cell on ne peut pas définir les cdl topologiqument

left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))

#bcs = [fem.dirichletbc(np.zeros((tdim,)), left_dofs, V),]

# --- CORRECTION POUR LES SOUS-ESPACES ---
# On effondre (collapse) les sous-espaces pour lire leur géométrie
V0, _ = V.sub(0).collapse()
V1, _ = V.sub(1).collapse()
V2, _ = V.sub(2).collapse()

# 1. On passe le TUPLE (sous-espace_global, espace_effondré) pour faire le lien
right_dofs = fem.locate_dofs_geometrical((V.sub(0), V0), lambda x: np.logical_and.reduce([np.isclose(x[0], largeur/2,atol=3e-2),np.isclose(x[1], largeur/2,atol=3e-2),np.isclose(x[2], 0.0)]))
# Pareil pour V1
right_dofsB = fem.locate_dofs_geometrical((V.sub(1), V1), lambda x: np.logical_and.reduce([np.isclose(x[0], largeur/2,atol=3e-2),np.isclose(x[1], largeur/2,atol=3e-2),np.isclose(x[2], 0.0)]))
# Pareil pour V2
right_dofsC = fem.locate_dofs_geometrical((V.sub(2), V2), lambda x: np.isclose(x[2], 0.0))

# --- Définition des conditions aux limites ---
bcs = [    
    # 2. On prend left_dofs[0] (les noeuds de V.sub(0)) et on applique sur V.sub(0)
    fem.dirichletbc(default_scalar_type(0.0), right_dofs[0], V.sub(0)),
    fem.dirichletbc(default_scalar_type(0.0), right_dofsB[0], V.sub(1)), 
    # Pareil pour l'axe Y
    fem.dirichletbc(default_scalar_type(0.0), right_dofsC[0], V.sub(2)), 
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
Nincr = 10
loading = np.linspace(0, -0.9*2000000, Nincr + 1) #Initialisation de la fissuration à 90 % du chargement max de -2000 kN loading divisée par l'aire de la section dans la formulation ufl.

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


import matplotlib.pyplot as plt

# ==========================================
# PREPARATION POUR L'EXTRACTION DE TAU_ZY
# ==========================================
# Coordonnées du point : on décale très légèrement L pour être sûr de tomber dans le volume
point_eval = np.array([[largeur - e_3DP, largeur - es, L ]], dtype=np.float64)

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
idx_tau_yz = list(components.keys()).index("sigma_xx")


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
    mesh_anim.cell_data[r"$\sigma_{x,x}$"] = np.zeros(mesh_anim.n_cells)
    # 4. Ajouter la maille au plotter avec une échelle adaptée à la physique (très petite !)
    clim_max = 1e6 #1e-9  # Amplitude maximale attendue pour c = -50
    # Configuration de la barre de couleur
    
    # 1. On vide le titre de la barre de couleur
    sargs = {
        "title": "",  # <-- Plus de titre ici !
        "label_font_size": 32,
        "position_x": 0.2,  # On place la barre à droite
        "height": 0.1,       # On réduit un peu la hauteur pour laisser de la place en haut
        "width": 0.6
    }

    # 2. On ajoute le titre de manière 100% indépendante
    titre_actor=plotter_anim.add_text(
        r"$\sigma_{x,x}$    ", 
        position=(0.5, 0.125), # 85% vers la droite, 90% vers le haut
        viewport=True,         # <-- Indispensable pour que PyVista comprenne les %
        font_size=32,
        name="tau_title"         # Le nommer permet de l'écraser si besoin dans la boucle
    )
    # 2. Centrer le point d'ancrage horizontalement et verticalement
    titre_actor.prop.justification_horizontal = 'center'
    #titre_actor.prop.justification_vertical = 'center'

    actor_anim = plotter_anim.add_mesh(
        mesh_anim, 
        scalars=r"$\sigma_{x,x}$", 
        cmap="jet", #"cmap="Reds", 
        show_edges=True,
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
    plotter_anim.open_gif("results/Compression_poutre.gif")
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
        val_locale = tau_func.x.array[192551] #192551 (valeur pour un enrobage de 3 cm) #300582 #360682 #192551  #.eval(points_on_proc, cells_for_eval)[0]
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
        mesh_anim.cell_data[r"$\sigma_{x,x}$"] =tau_array
        
        # OPTIONNEL : Ajuster dynamiquement l'échelle de couleur de la légende
        #plotter_anim.update_scalar_bar_range([0, max_real_disp if max_real_disp > 1e-15 else 1.0])
        # On récupère l'objet "barre de couleur" actif
        
        plotter_anim.add_text(
            f"Pas {i+1}/{len(loading)-1}"+ r"| $\sigma_{x,x}$ =" +f"{t * 1e-3:.0f} kN", 
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
# CRITÈRE GLOBAL ET LOCAL POUR L'ÉTUDE DE CONVERGENCE
# =====================================================================
# 1. Calcul de l'énergie de déformation totale
energy_form = 0.5 * ufl.inner(sigma(u), epsilon(u)) * dx
energy_compiled = fem.form(energy_form)
energy_local = fem.assemble_scalar(energy_compiled)

# 2. Calcul du déplacement maximal selon Z
# On extrait le sous-espace correspondant à l'axe Z (indice 2)
u_z = u.sub(2).collapse()
# On prend la valeur absolue maximale locale (pour capter compression ou traction)
max_uz_local = np.max(np.abs(u_z.x.array))

# 3. Synchronisation MPI (important si tu lances le code sur plusieurs cœurs)
energy_global = MPI.COMM_WORLD.allreduce(energy_local, op=MPI.SUM)
max_uz_global = MPI.COMM_WORLD.allreduce(max_uz_local, op=MPI.MAX)

# 4. Comptage des degrés de liberté (DDL)
num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

# 5. Affichage des résultats
if MPI.COMM_WORLD.rank == 0:
    print("\n" + "="*50)
    print(" BILAN DE CONVERGENCE DU MAILLAGE (CHARGE MAX)")
    print("="*50)
    print(f"Fichier maillage    : colonne.msh")
    print(f"Nombre de DDL       : {num_dofs_global}")
    print(f"Énergie totale      : {energy_global:.6e} Joules")
    print(f"Déplacement Z max   : {max_uz_global * 1000:.4f} mm")
    print("="*50 + "\n")


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
    
    
    print("L'animation 3D a été enregistrée avec succès dans 'results/torsion_poutre.gif'")




 

# ==========================================
# TRACÉ DE LA COURBE TAU_ZY = f(Couple)
# ==========================================
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    graphics_setup() 
    # On prend la valeur absolue de la charge si elle est négative (convention)
    loads_plot = np.abs(load_history)*1e-3  # Conversion en kN pour l'axe X
    tau_zy_history = np.abs(np.array(tau_zy_history)*1e-6)  # Conversion en tableau NumPy pour un tracé plus facile
    
    # =========================================================================
    # NOUVEAU : Sauvegarde des données dans un fichier CSV
    # =========================================================================
    # On empile les deux tableaux sous forme de colonnes
    data_to_save = np.column_stack((loads_plot, tau_zy_history))
    
    # Sauvegarde dans le dossier results (qui doit déjà exister)
    np.savetxt(
        "results/donnees_sigma_xx.csv", 
        data_to_save, 
        delimiter=",", 
        header="Force_kN,Sigma_xx_MPa", 
        comments="" # Permet de ne pas avoir de '#' au début du header
    )



    plt.plot(loads_plot, tau_zy_history, marker='o', linestyle='-', color='#d62728', linewidth=2)
    plt.xlabel("Force [kN]", fontsize=12)
    plt.ylabel(r"Traction $\sigma_{x,x}$ [MPa]", fontsize=12)
    #plt.title(r"Évolution de $\sigma_{x,x}$ au point" + f"({largeur - e_3DP:.2f}, {H/2:.2f}, {L:.2f})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0,1)
    # Set the limits
    plt.xlim(0, 2000)
    # Force the x-ticks to appear every 100 kN
    plt.xticks(np.arange(0, 2001, 300))
    
    # Sauvegarde et affichage
    plt.tight_layout()
    plt.savefig("results/courbe_sigma_xx.pdf", bbox_inches='tight')
    plt.show()

    