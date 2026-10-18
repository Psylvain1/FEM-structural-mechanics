import gmsh
import numpy as np

def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Poutre_Composite_Armee_Flambement")

    # ==========================================
    # Paramètres de base (Modifiés pour le Flambement)
    # ==========================================
    largeur = 0.24      # 24 cm : Réduite pour créer un axe de moindre inertie
    hauteur = 0.3       # 30 cm : Augmentée pour garder une section réaliste
    epaisseur_U = 0.04  # 4 cm : Épaisseur du coffrage sur les côtés
    longueur = 3.5      # 3.5 m : Augmentée pour exploser l'élancement géométrique
    U_bottom = epaisseur_U / 2 # 2 cm : Epaisseur du coffrage en bas
    cut=0.01

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
    espacement = 0.1         # Espacement entre chaque cadre sur Z (10 cm)
    
    # ------------------------------------------------------------------
    # COORDONNÉES DES ACIERS LONGITUDINAUX
    # ------------------------------------------------------------------
    x_gauche = epaisseur_U + 0.025 
    x_droite = largeur - epaisseur_U - 0.025 
    x_milieu = largeur / 2.0  
    
    y_bas = es  
    y_haut = hauteur - epaisseur_U - 0.02 

    # ------------------------------------------------------------------
    # NOUVEAU : COORDONNÉES DÉCALÉES DES CADRES (Ceinturage extérieur)
    # ------------------------------------------------------------------
    # On décale le centre des cadres vers l'extérieur : Rayon acier + Rayon cadre
    x_cadre_gauche = x_gauche - R - R_cadre
    x_cadre_droite = x_droite + R + R_cadre
    
    y_cadre_bas = y_bas - R - R_cadre
    y_cadre_haut = y_haut + R_haut + R_cadre  # Attention, on utilise R_haut ici
    
    w_cadre = x_cadre_droite - x_cadre_gauche
    h_cadre = y_cadre_haut - y_cadre_bas

    occ = gmsh.model.occ

    # ==========================================
    # 1. Création des volumes 3D 
    # ==========================================
    
    # --- Coffrage U ---
    box_global = occ.addBox(0, 0, 0, largeur, hauteur, longueur)
    box_vide = occ.addBox(epaisseur_U, U_bottom, epaisseur_U, largeur - 2*epaisseur_U, hauteur, longueur-2*epaisseur_U)
    #cut_vide = occ.addBox(0, 0, longueur/2-cut/2, largeur, U_bottom, cut)
    # 1. Création des 3 points du triangle dans le plan YZ (à X=0)
    p1 = occ.addPoint(0, 0, longueur/2 - cut/2)        # Bas gauche de l'entaille
    p2 = occ.addPoint(0, 0, longueur/2 + cut/2)        # Bas droite de l'entaille
    p3 = occ.addPoint(0, es-0.008, longueur/2)         # Pointe du triangle (sommet)

    # 2. Création des lignes reliant les points
    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p1)

    # 3. Création de la surface triangulaire
    cl = occ.addCurveLoop([l1, l2, l3])
    surf_triangle = occ.addPlaneSurface([cl])

    # 4. Extrusion de la surface le long de l'axe X pour en faire un volume 3D
    # L'extrusion renvoie une liste d'entités créées, on récupère celle de dimension 3 (le volume)
    extrude_out = occ.extrude([(2, surf_triangle)], largeur, 0, 0)
    cut_vide = [tag for dim, tag in extrude_out if dim == 3][0]
    # --- AJOUT : On crée une copie de cut_vide pour le noyau ---
    cut_vide_copie = occ.copy([(3, cut_vide)])[0]

    u_shape = occ.cut([(3, box_global)], [(3, box_vide),(3, cut_vide)])
    vol_U = u_shape[0][0]
    
    # --- Aciers longitudinaux --- 
    # 3 Aciers en bas
    acier1 = occ.addCylinder(x_gauche, y_bas, epaisseur_U, 0, 0, longueur-2*epaisseur_U, R)
    acier2 = occ.addCylinder(x_droite, y_bas, epaisseur_U, 0, 0, longueur-2*epaisseur_U, R)
    acier3 = occ.addCylinder(x_milieu, y_bas, epaisseur_U, 0, 0, longueur-2*epaisseur_U, R)

    # 2 Aciers en haut
    acier4 = occ.addCylinder(x_gauche, y_haut, epaisseur_U, 0, 0, longueur-2*epaisseur_U, R_haut)
    acier5 = occ.addCylinder(x_droite, y_haut, epaisseur_U, 0, 0, longueur-2*epaisseur_U, R_haut)

    # --- Cadres (Générés sur le périmètre extérieur) ---
    cadres_vols = []
    z_cadres = np.arange(espacement / 2.0, longueur, espacement)
    z_cadres = z_cadres[~np.isclose(z_cadres, 1.75)]
    
    for z in z_cadres:
        # Sphères de liaison (aux coins du cadre)
        s_bas_gauche = occ.addSphere(x_cadre_gauche, y_cadre_bas, z, R_cadre)
        s_bas_droite = occ.addSphere(x_cadre_droite, y_cadre_bas, z, R_cadre)
        s_haut_gauche = occ.addSphere(x_cadre_gauche, y_cadre_haut, z, R_cadre)
        s_haut_droite = occ.addSphere(x_cadre_droite, y_cadre_haut, z, R_cadre)        
        
        # Les 4 cylindres formant le cadre rectangulaire
        c_bas = occ.addCylinder(x_cadre_gauche, y_cadre_bas, z, w_cadre, 0, 0, R_cadre)
        c_haut = occ.addCylinder(x_cadre_gauche, y_cadre_haut, z, w_cadre, 0, 0, R_cadre)
        c_gauche = occ.addCylinder(x_cadre_gauche, y_cadre_bas, z, 0, h_cadre, 0, R_cadre)
        c_droite = occ.addCylinder(x_cadre_droite, y_cadre_bas, z, 0, h_cadre, 0, R_cadre)

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
    box_noyau = occ.addBox(epaisseur_U, U_bottom, epaisseur_U, largeur - 2*epaisseur_U, hauteur - U_bottom, longueur-2*epaisseur_U)
    #vol_Noyau = (3, box_noyau) 
    #TODO Enlever le volume du cutt
    
    # 2ème découpe avec la copie (qui sera détruite à son tour)
    u_shapeB = occ.cut([(3, box_noyau)], [cut_vide_copie])
    vol_Noyau = u_shapeB[0][0] 
    

    
    # ==========================================
    # 3. Fragmentation pour maillage conforme
    # ==========================================
    occ.removeAllDuplicates()
    
    # --- NOUVEAU : Création d'un plan de coupe exact à L/2 ---
    # Cela force le mailleur à placer des noeuds exactement au milieu de la poutre
    plane_mid = occ.addRectangle(0, 0, longueur / 2.0, largeur, hauteur)

    entities_to_fragment = [vol_U, vol_Noyau] + aciers_dimtags
    
    # On ajoute le plan (dim 2) à la liste des outils de fragmentation
    out, out_map = occ.fragment([entities_to_fragment[0]], entities_to_fragment[1:] + [(2, plane_mid)])
    occ.synchronize()

    # --- CORRECTION DE SÉCURITÉ ---
    # Comme on a fragmenté avec un plan 2D, il faut s'assurer de ne récupérer 
    # que les volumes (dim == 3) pour nos groupes physiques.
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
    
    # ==========================================
    # 4.5. Raffinement de maillage local (Mesh Fields)
    # ==========================================
    # 1. Création d'un champ de type 'Box' 
    gmsh.model.mesh.field.add("Box", 1)
    
    # 2. Tailles des éléments ciblées
    gmsh.model.mesh.field.setNumber(1, "VIn", 0.001)   # Très fin près de l'entaille (2 mm)
    gmsh.model.mesh.field.setNumber(1, "VOut", 0.035)  # Taille max ailleurs (35 mm)
    
    # 3. Délimitation spatiale de la zone de raffinement (autour de cut_vide)
    # On ajoute une 'marge' pour que la zone fine englobe bien l'entaille
    marge_z = 0.05
    marge_xy = 0.02
    
    gmsh.model.mesh.field.setNumber(1, "XMin", 0 - marge_xy)
    gmsh.model.mesh.field.setNumber(1, "XMax", largeur + marge_xy)
    
    gmsh.model.mesh.field.setNumber(1, "YMin", 0 - marge_xy)
    gmsh.model.mesh.field.setNumber(1, "YMax", epaisseur_U + marge_xy)
    
    # Centré sur l'entaille en Z (longueur/2)
    gmsh.model.mesh.field.setNumber(1, "ZMin", (longueur/2 - cut/2) - marge_z)
    gmsh.model.mesh.field.setNumber(1, "ZMax", (longueur/2 + cut/2) + marge_z)
    
    # 4. Épaisseur de la zone de transition (pour éviter un saut brusque de taille)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 0.03)
    
    # 5. Définir ce champ comme le contrôleur principal de la taille du maillage
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    
    # Empêcher les tailles globales (MeshSizeMax/Min) d'écraser le champ de fond
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1) # Garder la courbure pour les aciers
    
    # ==========================================
    # 5. Paramètres du maillage et Génération
    # ==========================================
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-5)
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005) 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.035)
    
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12) 
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)   
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
    gmsh.option.setNumber("Mesh.Optimize", 1)

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write("poutre.msh")

    # ==========================================
    # 6. Configuration de la vue en coupe (L/3)
    # ==========================================
    # Équation du plan 0 : A*x + B*y + C*z + D = 0
    # Coupe à z = longueur / 3  =>  1*z - (longueur/3) = 0
    gmsh.option.setNumber("General.Clip0A", 0)
    gmsh.option.setNumber("General.Clip0B", 0)
    gmsh.option.setNumber("General.Clip0C", 1) 
    gmsh.option.setNumber("General.Clip0D", -longueur / 3.0)

    # NOUVELLE SYNTAXE (Gmsh 4+) : 
    # On active le plan n°0 (valeur = 2^0 = 1) pour la géométrie et le maillage
    gmsh.option.setNumber("Geometry.Clip", 1)
    gmsh.option.setNumber("Mesh.Clip", 1)
    
    # Option d'affichage pour le maillage :
    # 0 = Coupe "lisse" géométrique au milieu des éléments
    # 1 = Retire complètement les éléments touchés par la coupe
    gmsh.option.setNumber("General.ClipWholeElements", 0)
    
    #gmsh.fltk.run() 
    gmsh.finalize()
    
    return largeur, hauteur, epaisseur_U, longueur, es, phi

if __name__ == "__main__":
    create_geometry()
