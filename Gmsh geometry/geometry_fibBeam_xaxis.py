import gmsh
import numpy as np

def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Poutre_Composite_Armee_Flambement")

    # ==========================================
    # Paramètres de base (Modifiés pour le Flambement)
    # ==========================================
    largeur = 0.24      # 24 cm : Réduite pour créer un axe de moindre inertie (maintenant sur Z)
    hauteur = 0.3       # 30 cm : Augmentée pour garder une section réaliste (sur Y)
    epaisseur_U = 0.04  # 4 cm : Épaisseur du coffrage sur les côtés
    longueur = 3.5      # 3.5 m : Augmentée pour exploser l'élancement géométrique (maintenant sur X)
    U_bottom = epaisseur_U / 2 # 2 cm : Epaisseur du coffrage en bas

    # Aciers du bas
    es = U_bottom + 0.02      # Centre Y des aciers longitudinaux bas
    phi = 0.016               # Diamètre des aciers du bas (16 mm)
    R = phi / 2.0

    # Aciers du haut
    phi_haut = 0.008          # Diamètre des aciers du haut (8 mm)
    R_haut = phi_haut / 2.0

    # ==========================================
    # Paramètres des cadres
    # ==========================================
    phi_cadre = 0.008        # Diamètre du cadre (8 mm)
    R_cadre = phi_cadre / 2.0
    espacement = 0.1         # Espacement entre chaque cadre sur X (10 cm)
    
    # ------------------------------------------------------------------
    # COORDONNÉES DES ACIERS LONGITUDINAUX (X devient Z)
    # ------------------------------------------------------------------
    z_gauche = epaisseur_U + 0.025 
    z_droite = largeur - epaisseur_U - 0.025 
    z_milieu = largeur / 2.0  
    
    y_bas = es  
    y_haut = hauteur - epaisseur_U - 0.02 

    # ------------------------------------------------------------------
    # NOUVEAU : COORDONNÉES DÉCALÉES DES CADRES (Ceinturage extérieur)
    # ------------------------------------------------------------------
    # On décale le centre des cadres vers l'extérieur : Rayon acier + Rayon cadre
    z_cadre_gauche = z_gauche - R - R_cadre
    z_cadre_droite = z_droite + R + R_cadre
    
    y_cadre_bas = y_bas - R - R_cadre
    y_cadre_haut = y_haut + R_haut + R_cadre  # Attention, on utilise R_haut ici
    
    w_cadre = z_cadre_droite - z_cadre_gauche
    h_cadre = y_cadre_haut - y_cadre_bas

    occ = gmsh.model.occ

    # ==========================================
    # 1. Création des volumes 3D 
    # ==========================================
    
    # --- Coffrage U ---
    box_global = occ.addBox(0, 0, 0, longueur, hauteur, largeur)
    box_vide = occ.addBox(epaisseur_U, U_bottom, epaisseur_U, longueur - 2*epaisseur_U, hauteur, largeur - 2*epaisseur_U)
    u_shape = occ.cut([(3, box_global)], [(3, box_vide)])
    vol_U = u_shape[0][0]
    
    # --- Aciers longitudinaux --- 
    # 3 Aciers en bas
    acier1 = occ.addCylinder(epaisseur_U, y_bas, z_gauche, longueur-2*epaisseur_U, 0, 0, R)
    acier2 = occ.addCylinder(epaisseur_U, y_bas, z_droite, longueur-2*epaisseur_U, 0, 0, R)
    acier3 = occ.addCylinder(epaisseur_U, y_bas, z_milieu, longueur-2*epaisseur_U, 0, 0, R)

    # 2 Aciers en haut
    acier4 = occ.addCylinder(epaisseur_U, y_haut, z_gauche, longueur-2*epaisseur_U, 0, 0, R_haut)
    acier5 = occ.addCylinder(epaisseur_U, y_haut, z_droite, longueur-2*epaisseur_U, 0, 0, R_haut)

    # --- Cadres (Générés sur le périmètre extérieur) ---
    cadres_vols = []
    x_cadres = np.arange(espacement / 2.0, longueur, espacement)
    
    for x in x_cadres:
        # Sphères de liaison (aux coins du cadre)
        s_bas_gauche = occ.addSphere(x, y_cadre_bas, z_cadre_gauche, R_cadre)
        s_bas_droite = occ.addSphere(x, y_cadre_bas, z_cadre_droite, R_cadre)
        s_haut_gauche = occ.addSphere(x, y_cadre_haut, z_cadre_gauche, R_cadre)
        s_haut_droite = occ.addSphere(x, y_cadre_haut, z_cadre_droite, R_cadre)        
        
        # Les 4 cylindres formant le cadre rectangulaire (étendus sur Y et Z)
        c_bas = occ.addCylinder(x, y_cadre_bas, z_cadre_gauche, 0, 0, w_cadre, R_cadre)
        c_haut = occ.addCylinder(x, y_cadre_haut, z_cadre_gauche, 0, 0, w_cadre, R_cadre)
        c_gauche = occ.addCylinder(x, y_cadre_bas, z_cadre_gauche, 0, h_cadre, 0, R_cadre)
        c_droite = occ.addCylinder(x, y_cadre_bas, z_cadre_droite, 0, h_cadre, 0, R_cadre)

        cadres_vols.extend([
            (3, s_bas_gauche), (3, s_bas_droite), (3, s_haut_gauche), (3, s_haut_droite),
            (3, c_bas), (3, c_haut), (3, c_gauche), (3, c_droite)
        ])

    # ==========================================
    # 2. Fusions et découpes booléennes
    # ==========================================
    
    # Fusion de tous les aciers
    acier_tools = [(3, acier2), (3, acier3), (3, acier4), (3, acier5)] + cadres_vols
    steel_fuse = occ.fuse([(3, acier1)], acier_tools, removeTool=True)
    aciers_dimtags = steel_fuse[0] 

    # --- Noyau Coulé ---
    box_noyau = occ.addBox(epaisseur_U, U_bottom, epaisseur_U, longueur - 2*epaisseur_U, hauteur - U_bottom, largeur - 2*epaisseur_U)
    vol_Noyau = (3, box_noyau) 

    # ==========================================
    # 3. Fragmentation pour maillage conforme
    # ==========================================
    occ.removeAllDuplicates()
    
    # --- Création d'un plan de coupe exact à L/2 dans le plan YZ ---
    # Nous définissons manuellement la surface pour qu'elle soit orthogonale à l'axe X
    p_pl1 = occ.addPoint(longueur / 2.0, 0, 0)
    p_pl2 = occ.addPoint(longueur / 2.0, hauteur, 0)
    p_pl3 = occ.addPoint(longueur / 2.0, hauteur, largeur)
    p_pl4 = occ.addPoint(longueur / 2.0, 0, largeur)
    
    l_pl1 = occ.addLine(p_pl1, p_pl2)
    l_pl2 = occ.addLine(p_pl2, p_pl3)
    l_pl3 = occ.addLine(p_pl3, p_pl4)
    l_pl4 = occ.addLine(p_pl4, p_pl1)
    
    cl_pl = occ.addCurveLoop([l_pl1, l_pl2, l_pl3, l_pl4])
    plane_mid = occ.addPlaneSurface([cl_pl])

    entities_to_fragment = [vol_U, vol_Noyau] + aciers_dimtags
    
    # On ajoute le plan (dim 2) à la liste des outils de fragmentation
    out, out_map = occ.fragment([entities_to_fragment[0]], entities_to_fragment[1:] + [(2, plane_mid)])
    occ.synchronize()

    # --- CORRECTION DE SÉCURITÉ ---
    tags_U = [tag for dim, tag in out_map[0] if dim == 3]
    tags_Noyau = [tag for dim, tag in out_map[1] if dim == 3]
    
    tags_Aciers = []
    for i in range(2, len(out_map)):
        for dim, tag in out_map[i]:
            if dim == 3:  # <-- Très important pour ignorer les restes du plan 2D
                tags_Aciers.append(tag)
    tags_Aciers = list(set(tags_Aciers))

    # ==========================================
    # 4. Groupes Physiques
    # ==========================================
    gmsh.model.addPhysicalGroup(3, tags_U, name="Coffrage_U", tag=1)
    gmsh.model.addPhysicalGroup(3, tags_Noyau, name="Coeur_Coule", tag=2)
    gmsh.model.addPhysicalGroup(3, tags_Aciers, name="Aciers", tag=3)

    # ------------------------------------------------------------------
    # NOUVEAU : Extraction de l'interface (Surfaces 2D)
    # ------------------------------------------------------------------
    
    # 1. Obtenir toutes les surfaces délimitant le coffrage U
    surfaces_U = set()
    for vol in tags_U:
        _, adj_surfaces = gmsh.model.getAdjacencies(3, vol)
        surfaces_U.update(adj_surfaces)

    # 2. Obtenir toutes les surfaces délimitant le noyau
    surfaces_Noyau = set()
    for vol in tags_Noyau:
        _, adj_surfaces = gmsh.model.getAdjacencies(3, vol)
        surfaces_Noyau.update(adj_surfaces)

    # 3. L'interface est l'intersection exacte des deux ensembles de surfaces
    interface_tags = list(surfaces_U.intersection(surfaces_Noyau))

    # 4. Création du groupe physique 2D
    gmsh.model.addPhysicalGroup(2, interface_tags, name="Interface_Coffrage_Noyau", tag=4)
    

    # ==========================================
    # 5. Paramètres du maillage et Génération
    # ==========================================

    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-5)
    
    # Prevent Gmsh from pushing boundary sizes too far into the volume
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Global min/max as safety nets (Fields will do the actual work)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005) # 5 mm 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)   # 10 cm

    # Pour une étude de convergence, vous pouvez passer ce facteur de 1.0 à 0.5, puis 0.25, etc.
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.8)

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 5)    

    # Use standard 3D algorithms (Delaunay is good, HXT is often faster for multithreading)
    gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay for 3D
    gmsh.option.setNumber("Mesh.Optimize", 1)
    
    gmsh.model.mesh.generate(3)
    gmsh.write("poutre.msh")
    
    #gmsh.fltk.run() 
    gmsh.finalize()
    
    return largeur, hauteur, epaisseur_U, longueur, es, phi

if __name__ == "__main__":
    create_geometry()