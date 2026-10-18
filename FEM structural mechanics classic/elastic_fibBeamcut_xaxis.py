import gmsh
import numpy as np

def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Poutre_Composite_Armee_Flambement")

    # ==========================================
    # Paramètres de base (Modifiés pour le Flambement)
    # ==========================================
    largeur = 0.24      # 24 cm : Réduite pour créer un axe de moindre inertie (Maintenant sur Z)
    hauteur = 0.3       # 30 cm : Augmentée pour garder une section réaliste (Reste sur Y)
    epaisseur_U = 0.04  # 4 cm : Épaisseur du coffrage sur les côtés
    longueur = 3.5      # 3.5 m : Augmentée pour exploser l'élancement géométrique (Maintenant sur X)
    U_bottom = epaisseur_U / 2 # 2 cm : Epaisseur du coffrage en bas
    cut = 0.01

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
    # COORDONNÉES DES ACIERS LONGITUDINAUX (Anciennement X, maintenant Z)
    # ------------------------------------------------------------------
    z_gauche = epaisseur_U + 0.025 
    z_droite = largeur - epaisseur_U - 0.025 
    z_milieu = largeur / 2.0  
    
    y_bas = es  
    y_haut = hauteur - epaisseur_U - 0.02 

    # ------------------------------------------------------------------
    # COORDONNÉES DÉCALÉES DES CADRES (Ceinturage extérieur)
    # ------------------------------------------------------------------
    z_cadre_gauche = z_gauche - R - R_cadre
    z_cadre_droite = z_droite + R + R_cadre
    
    y_cadre_bas = y_bas - R - R_cadre
    y_cadre_haut = y_haut + R_haut + R_cadre
    
    w_cadre = z_cadre_droite - z_cadre_gauche # Largeur du cadre sur Z
    h_cadre = y_cadre_haut - y_cadre_bas      # Hauteur du cadre sur Y

    occ = gmsh.model.occ

    # ==========================================
    # 1. Création des volumes 3D 
    # ==========================================
    
    # --- Coffrage U ---
    # Coordonnées : x, y, z, dx, dy, dz
    box_global = occ.addBox(0, 0, 0, longueur, hauteur, largeur)
    box_vide = occ.addBox(epaisseur_U, U_bottom, epaisseur_U, longueur - 2*epaisseur_U, hauteur, largeur - 2*epaisseur_U)
    
    # 1. Création des 3 points du triangle dans le plan XY (à Z=0)
    p1 = occ.addPoint(longueur/2 - cut/2, 0, 0)        # Bas gauche de l'entaille
    p2 = occ.addPoint(longueur/2 + cut/2, 0, 0)        # Bas droite de l'entaille
    p3 = occ.addPoint(longueur/2, es-0.008, 0)         # Pointe du triangle (sommet)

    # 2. Création des lignes reliant les points
    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p1)

    # 3. Création de la surface triangulaire
    cl = occ.addCurveLoop([l1, l2, l3])
    surf_triangle = occ.addPlaneSurface([cl])

    # 4. Extrusion de la surface le long de l'axe Z pour en faire un volume 3D
    extrude_out = occ.extrude([(2, surf_triangle)], 0, 0, largeur)
    cut_vide = [tag for dim, tag in extrude_out if dim == 3][0]
    
    # --- AJOUT : On crée une copie de cut_vide pour le noyau ---
    cut_vide_copie = occ.copy([(3, cut_vide)])[0]

    u_shape = occ.cut([(3, box_global)], [(3, box_vide),(3, cut_vide)])
    vol_U = u_shape[0][0]
    
    # --- Aciers longitudinaux (S'étendent selon X) --- 
    # 3 Aciers en bas
    acier1 = occ.addCylinder(epaisseur_U, y_bas, z_gauche, longueur-2*epaisseur_U, 0, 0, R)
    acier2 = occ.addCylinder(epaisseur_U, y_bas, z_droite, longueur-2*epaisseur_U, 0, 0, R)
    acier3 = occ.addCylinder(epaisseur_U, y_bas, z_milieu, longueur-2*epaisseur_U, 0, 0, R)

    # 2 Aciers en haut
    acier4 = occ.addCylinder(epaisseur_U, y_haut, z_gauche, longueur-2*epaisseur_U, 0, 0, R_haut)
    acier5 = occ.addCylinder(epaisseur_U, y_haut, z_droite, longueur-2*epaisseur_U, 0, 0, R_haut)

    # --- Cadres (Générés sur le périmètre extérieur le long de l'axe X) ---
    cadres_vols = []
    x_cadres = np.arange(espacement / 2.0, longueur, espacement)
    x_cadres = x_cadres[~np.isclose(x_cadres, 1.75)]
    
    for x_pos in x_cadres:
        # Sphères de liaison (aux coins du cadre)
        s_bas_gauche = occ.addSphere(x_pos, y_cadre_bas, z_cadre_gauche, R_cadre)
        s_bas_droite = occ.addSphere(x_pos, y_cadre_bas, z_cadre_droite, R_cadre)
        s_haut_gauche = occ.addSphere(x_pos, y_cadre_haut, z_cadre_gauche, R_cadre)
        s_haut_droite = occ.addSphere(x_pos, y_cadre_haut, z_cadre_droite, R_cadre)        
        
        # Les 4 cylindres formant le cadre rectangulaire (sur plan YZ)
        c_bas = occ.addCylinder(x_pos, y_cadre_bas, z_cadre_gauche, 0, 0, w_cadre, R_cadre)
        c_haut = occ.addCylinder(x_pos, y_cadre_haut, z_cadre_gauche, 0, 0, w_cadre, R_cadre)
        c_gauche = occ.addCylinder(x_pos, y_cadre_bas, z_cadre_gauche, 0, h_cadre, 0, R_cadre)
        c_droite = occ.addCylinder(x_pos, y_cadre_bas, z_cadre_droite, 0, h_cadre, 0, R_cadre)

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
    
    # 2ème découpe avec la copie
    u_shapeB = occ.cut([(3, box_noyau)], [cut_vide_copie])
    vol_Noyau = u_shapeB[0][0] 
    
    # ==========================================
    # 3. Fragmentation pour maillage conforme
    # ==========================================
    occ.removeAllDuplicates()
    
    # --- Création d'un plan de coupe exact à L/2 dans le plan YZ ---
    p_mid1 = occ.addPoint(longueur / 2.0, 0, 0)
    p_mid2 = occ.addPoint(longueur / 2.0, hauteur, 0)
    p_mid3 = occ.addPoint(longueur / 2.0, hauteur, largeur)
    p_mid4 = occ.addPoint(longueur / 2.0, 0, largeur)
    l_mid1 = occ.addLine(p_mid1, p_mid2)
    l_mid2 = occ.addLine(p_mid2, p_mid3)
    l_mid3 = occ.addLine(p_mid3, p_mid4)
    l_mid4 = occ.addLine(p_mid4, p_mid1)
    cl_mid = occ.addCurveLoop([l_mid1, l_mid2, l_mid3, l_mid4])
    plane_mid = occ.addPlaneSurface([cl_mid])

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
            if dim == 3:  
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
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", 0.001)   
    gmsh.model.mesh.field.setNumber(1, "VOut", 0.035)  
    
    marge_x = 0.05
    marge_yz = 0.02
    
    # Sur l'axe Z (largeur)
    gmsh.model.mesh.field.setNumber(1, "ZMin", 0 - marge_yz)
    gmsh.model.mesh.field.setNumber(1, "ZMax", largeur + marge_yz)
    
    # Sur l'axe Y (hauteur/épaisseur)
    gmsh.model.mesh.field.setNumber(1, "YMin", 0 - marge_yz)
    gmsh.model.mesh.field.setNumber(1, "YMax", epaisseur_U + marge_yz)
    
    # Sur l'axe X (longueur), centré sur l'entaille
    gmsh.model.mesh.field.setNumber(1, "XMin", (longueur/2 - cut/2) - marge_x)
    gmsh.model.mesh.field.setNumber(1, "XMax", (longueur/2 + cut/2) + marge_x)
    
    gmsh.model.mesh.field.setNumber(1, "Thickness", 0.03)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1) 
    
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
    # Coupe à x = longueur / 3  =>  1*x - (longueur/3) = 0
    gmsh.option.setNumber("General.Clip0A", 1) # Axe X
    gmsh.option.setNumber("General.Clip0B", 0)
    gmsh.option.setNumber("General.Clip0C", 0) 
    gmsh.option.setNumber("General.Clip0D", -longueur / 3.0)

    gmsh.option.setNumber("Geometry.Clip", 1)
    gmsh.option.setNumber("Mesh.Clip", 1)
    gmsh.option.setNumber("General.ClipWholeElements", 0)
    
    # gmsh.fltk.run() 
    gmsh.finalize()
    
    return largeur, hauteur, epaisseur_U, longueur, es, phi

if __name__ == "__main__":
    create_geometry()