import gmsh
import numpy as np

def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Colonne_Composite_Armee")

    # ==========================================
    # Paramètres de base de la Colonne
    # ==========================================
    largeur = 0.34      # 34 cm : Largeur totale
    hauteur = 0.34      # 34 cm : Hauteur totale (section carrée)
    epaisseur_U = 0.04  # 4 cm : Épaisseur du coffrage périphérique (tout le tour)
    longueur = 1.0      # 1 m : Hauteur totale de la colonne sur l'axe Z

    # Aciers longitudinaux (4 aciers)
    espacement_paroi = 0.03   # 3 cm d'espacement de la paroi interne
    phi = 0.016               # Diamètre des aciers longitudinaux (16 mm par défaut)
    R = phi / 2.0

    # ==========================================
    # Paramètres des cadres
    # ==========================================
    phi_cadre = 0.008        # Diamètre du cadre (8 mm)
    R_cadre = phi_cadre / 2.0
    espacement = 0.225         # Espacement entre chaque cadre sur l'axe Z (10 cm)
    a_tip=0.05 # 5 cm : distance entre d'arrêt des armatures longitudinaux.
    
    # ------------------------------------------------------------------
    # COORDONNÉES DES ACIERS LONGITUDINAUX
    # ------------------------------------------------------------------
    # Calcul par rapport à l'extérieur du coffrage (qui est le repère 0,0)
    x_gauche = epaisseur_U + espacement_paroi
    x_droite = largeur - epaisseur_U - espacement_paroi
    
    y_bas = epaisseur_U + espacement_paroi
    y_haut = hauteur - epaisseur_U - espacement_paroi

    # ------------------------------------------------------------------
    # COORDONNÉES DÉCALÉES DES CADRES (Ceinturage extérieur)
    # ------------------------------------------------------------------
    # On décale le centre des cadres vers l'extérieur : Rayon acier + Rayon cadre
    x_cadre_gauche = x_gauche - R - R_cadre
    x_cadre_droite = x_droite + R + R_cadre
    
    y_cadre_bas = y_bas - R - R_cadre
    y_cadre_haut = y_haut + R + R_cadre 
    
    w_cadre = x_cadre_droite - x_cadre_gauche
    h_cadre = y_cadre_haut - y_cadre_bas

    occ = gmsh.model.occ

    # ==========================================
    # 1. Création des volumes 3D 
    # ==========================================
    
    # --- Coffrage Imprimé (Tube carré) ---
    box_global = occ.addBox(0, 0, 0, largeur, hauteur, longueur)
    # Le vide traverse toute la longueur (Z de 0 à la longueur totale)
    box_vide = occ.addBox(epaisseur_U, epaisseur_U, 0, largeur - 2*epaisseur_U, hauteur - 2*epaisseur_U, longueur)
    
    # Découpe pour créer le coffrage périphérique
    u_shape = occ.cut([(3, box_global)], [(3, box_vide)])
    vol_U = u_shape[0][0]
    
    # --- Aciers longitudinaux (Les 4 coins) --- 
    acier1 = occ.addCylinder(x_gauche, y_bas, a_tip, 0, 0, longueur-2*a_tip, R)
    acier2 = occ.addCylinder(x_droite, y_bas, a_tip, 0, 0, longueur-2*a_tip, R)
    acier3 = occ.addCylinder(x_gauche, y_haut, a_tip, 0, 0, longueur-2*a_tip, R)
    acier4 = occ.addCylinder(x_droite, y_haut, a_tip, 0, 0, longueur-2*a_tip, R)

    # --- Cadres (Générés sur le périmètre extérieur des aciers) ---
    cadres_vols = []
    z_cadres = np.arange(a_tip+espacement, longueur - a_tip, espacement)
    print(z_cadres)
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
    acier_tools = [(3, acier2), (3, acier3), (3, acier4)] + cadres_vols
    steel_fuse = occ.fuse([(3, acier1)], acier_tools, removeTool=True)
    aciers_dimtags = steel_fuse[0] 

    # --- Noyau Coulé ---
    box_noyau = occ.addBox(epaisseur_U, epaisseur_U, 0, largeur - 2*epaisseur_U, hauteur - 2*epaisseur_U, longueur)
    vol_Noyau = (3, box_noyau) 

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
    gmsh.model.addPhysicalGroup(3, tags_U, name="Coffrage_Imprime", tag=1)
    gmsh.model.addPhysicalGroup(3, tags_Noyau, name="Coeur_Coule", tag=2)
    gmsh.model.addPhysicalGroup(3, tags_Aciers, name="Aciers", tag=3)

    # ------------------------------------------------------------------
    # NOUVEAU : Extraction de l'interface (Surfaces 2D)
    # ------------------------------------------------------------------
    
    # 1. Obtenir toutes les surfaces délimitant le coffrage U
    surfaces_U = set()
    for vol in tags_U:
        # getAdjacencies(3, vol) renvoie (adjacences_superieures, adjacences_inferieures)
        # Pour un volume (dim 3), les adjacences inférieures sont ses surfaces (dim 2)
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
    # J'utilise tag=4 pour éviter le conflit avec le Coeur_Coule (tag=2)
    gmsh.model.addPhysicalGroup(2, interface_tags, name="Interface_Coffrage_Noyau", tag=4)
    

    # ==========================================
    # 5. Paramètres du maillage et Génération
    # ==========================================
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-5)

    # Pour une étude de convergence, vous pouvez passer ce facteur de 1.0 à 0.5, puis 0.25, etc.
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005) 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.035)
    
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12) 
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)   
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
    gmsh.option.setNumber("Mesh.Optimize", 1)

    # Dé-commentez pour générer automatiquement le maillage au lancement
    gmsh.model.mesh.generate(3)
    gmsh.write("colonne.msh")
    
    #gmsh.fltk.run() 
    gmsh.finalize()
    
    return largeur, hauteur, epaisseur_U, longueur, espacement_paroi, phi

if __name__ == "__main__":
    create_geometry()

