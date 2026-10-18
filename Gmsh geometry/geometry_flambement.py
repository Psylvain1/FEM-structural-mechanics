import gmsh
import numpy as np

def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Poutre_Composite_Armee_Flambement")

    # ==========================================
    # Paramètres de base (Modifiés pour le Flambement)
    # ==========================================
    largeur = 0.14      # 14 cm : Réduite pour créer un axe de moindre inertie (axe faible)
    hauteur = 0.25      # 25 cm : Augmentée pour garder une section réaliste
    epaisseur_U = 0.03  # 3 cm : Épaisseur du coffrage
    longueur = 7.5      # 7.5 m : Augmentée pour exploser l'élancement géométrique

    es = epaisseur_U + 0.03  # Centre Y des aciers longitudinaux (6 cm du bas)
    phi = 0.01               # Diamètre des aciers longitudinaux (10 mm)
    R = phi / 2.0

    # ==========================================
    # Paramètres des cadres
    # ==========================================
    phi_cadre = 0.008        # Diamètre du cadre (8 mm)
    R_cadre = phi_cadre / 2.0
    espacement = 0.2         # Espacement entre chaque cadre sur Z (20 cm)
    
    # ------------------------------------------------------------------
    # CORRECTION CRUCIALE : INTERSECTIONS FRANCHES ET PARAMÉTRIQUES
    # Les coordonnées des cadres et des aciers sont désormais verrouillées
    # ensemble pour éviter les bugs booléens si on change la largeur.
    # ------------------------------------------------------------------
    # On place les aciers à 2.5 cm des bords internes du coffrage
    x_gauche = epaisseur_U + 0.025 
    x_droite = largeur - epaisseur_U - 0.025 
    w_cadre = x_droite - x_gauche
    
    y_bas = es  # Intersection parfaite en Y avec l'acier longitudinal
    y_haut = hauteur - epaisseur_U - 0.02 
    h_cadre = y_haut - y_bas

    occ = gmsh.model.occ

    # ==========================================
    # 1. Création des volumes 3D 
    # ==========================================
    
    # --- Coffrage U ---
    box_global = occ.addBox(0, 0, 0, largeur, hauteur, longueur)
    box_vide = occ.addBox(epaisseur_U, epaisseur_U, 0, largeur - 2*epaisseur_U, hauteur, longueur)
    u_shape = occ.cut([(3, box_global)], [(3, box_vide)])
    vol_U = u_shape[0][0]
    
    # --- Aciers longitudinaux (Coordonnées alignées avec les cadres) --- 
    acier1 = occ.addCylinder(x_gauche, y_bas, 0, 0, 0, longueur, R)
    acier2 = occ.addCylinder(x_droite, y_bas, 0, 0, 0, longueur, R)

    # --- Cadres (4 cylindres + 4 sphères de liaison) ---
    cadres_vols = []
    z_cadres = np.arange(espacement / 2.0, longueur, espacement)
    
    for z in z_cadres:
        # Sphères de liaison
        s_bas_gauche = occ.addSphere(x_gauche, y_bas, z, R_cadre)
        s_bas_droite = occ.addSphere(x_droite, y_bas, z, R_cadre)
        s_haut_gauche = occ.addSphere(x_gauche, y_haut, z, R_cadre)
        s_haut_droite = occ.addSphere(x_droite, y_haut, z, R_cadre)        
        
        # Les 4 cylindres du cadre
        c_bas = occ.addCylinder(x_gauche, y_bas, z, w_cadre, 0, 0, R_cadre)
        c_haut = occ.addCylinder(x_gauche, y_haut, z, w_cadre, 0, 0, R_cadre)
        c_gauche = occ.addCylinder(x_gauche, y_bas, z, 0, h_cadre, 0, R_cadre)
        c_droite = occ.addCylinder(x_droite, y_bas, z, 0, h_cadre, 0, R_cadre)

        cadres_vols.extend([
            (3, s_bas_gauche), (3, s_bas_droite), (3, s_haut_gauche), (3, s_haut_droite),
            (3, c_bas), (3, c_haut), (3, c_gauche), (3, c_droite)
        ])

    # ==========================================
    # 2. Fusions et découpes booléennes
    # ==========================================
    
    # Fusion de tous les aciers
    acier_tools = [(3, acier2)] + cadres_vols
    steel_fuse = occ.fuse([(3, acier1)], acier_tools, removeTool=True)
    aciers_dimtags = steel_fuse[0] 

    # --- Noyau Coulé ---
    box_noyau = occ.addBox(epaisseur_U, epaisseur_U, 0, largeur - 2*epaisseur_U, hauteur - epaisseur_U, longueur)
    vol_Noyau = (3, box_noyau) 

    # ==========================================
    # 3. Fragmentation pour maillage conforme
    # ==========================================
    occ.removeAllDuplicates()
    
    # On fragmente tout d'un coup (Coffrage, Noyau Brut, Aciers)
    entities_to_fragment = [vol_U, vol_Noyau] + aciers_dimtags
    out, out_map = occ.fragment([entities_to_fragment[0]], entities_to_fragment[1:])
    occ.synchronize()

    tags_U = [tag for dim, tag in out_map[0]]
    tags_Noyau = [tag for dim, tag in out_map[1]]
    
    tags_Aciers = []
    for i in range(2, len(out_map)):
        for dim, tag in out_map[i]:
            tags_Aciers.append(tag)
    tags_Aciers = list(set(tags_Aciers))

    # ==========================================
    # 4. Groupes Physiques
    # ==========================================
    gmsh.model.addPhysicalGroup(3, tags_U, name="Coffrage_U", tag=1)
    gmsh.model.addPhysicalGroup(3, tags_Noyau, name="Coeur_Coule", tag=2)
    gmsh.model.addPhysicalGroup(3, tags_Aciers, name="Aciers", tag=3)

    # ==========================================
    # 5. Paramètres du maillage et Génération
    # ==========================================
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-5)
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005) 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.035)
    
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12) 
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)

    gmsh.model.mesh.generate(3)
    gmsh.write("poutre.msh")
    
    gmsh.fltk.run() 
    gmsh.finalize()
    
    return largeur, hauteur, epaisseur_U, longueur, es, phi

if __name__ == "__main__":
    create_geometry()