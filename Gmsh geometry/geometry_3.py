import gmsh
import numpy as np

def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Poutre_Composite_Armee")

    # ==========================================
    # Paramètres de base
    # ==========================================
    largeur = 0.2
    hauteur = 0.2
    epaisseur_U = 0.03
    longueur = 2.0

    es = epaisseur_U + 0.01  # Enrobage (hauteur du centre des aciers long.)
    phi = 0.01               # Diamètre des aciers longitudinaux
    R = phi / 2.0

    # ==========================================
    # NOUVEAU : Paramètres des cadres 
    # ==========================================
    phi_cadre = 0.008        # Épaisseur d'un cadre (8 mm)
    espacement = 0.2         # Espacement entre chaque cadre sur Z (20 cm)
    
    # Dimensions calculées pour que le cadre englobe les aciers longitudinaux
    w_cadre = 6 * phi
    h_cadre = hauteur - epaisseur_U - 0.012
    x_cadre = largeur / 2.0 - 3 * phi
    y_cadre = epaisseur_U + 0.002 # Léger retrait pour ne pas fusionner avec le Coffrage U

    occ = gmsh.model.occ

    # ==========================================
    # 1. Création des volumes 3D 
    # ==========================================
    
    # --- Coffrage U ---
    box_global = occ.addBox(0, 0, 0, largeur, hauteur, longueur)
    box_vide = occ.addBox(epaisseur_U, epaisseur_U, 0, largeur - 2*epaisseur_U, hauteur, longueur)
    u_shape = occ.cut([(3, box_global)], [(3, box_vide)])
    vol_U = u_shape[0][0]

    # --- Aciers longitudinaux ---
    acier1 = occ.addCylinder(largeur/2 - 2*phi, es, 0, 0, 0, longueur, R)
    acier2 = occ.addCylinder(largeur/2 + 2*phi, es, 0, 0, 0, longueur, R)

    # --- Cadres (Génération procédurale) ---
    cadres_vols = []
    # On positionne un cadre tous les 'espacement' sur la longueur de la poutre
    z_cadres = np.arange(espacement / 2, longueur, espacement)
    
    for z in z_cadres:
        # Cadre plein extérieur
        c_ext = occ.addBox(x_cadre, y_cadre, z - phi_cadre/2, w_cadre, h_cadre, phi_cadre)
        # Vide intérieur
        c_int = occ.addBox(x_cadre + phi_cadre, y_cadre + phi_cadre, z - phi_cadre/2, 
                           w_cadre - 2*phi_cadre, h_cadre - 2*phi_cadre, phi_cadre)
        # Soustraction pour obtenir un cadre creux
        cadre = occ.cut([(3, c_ext)], [(3, c_int)])
        cadres_vols.append(cadre[0][0])

    # ==========================================
    # 2. Fusions et découpes booléennes
    # ==========================================
    
    # Fusionner TOUS les aciers (longitudinaux + cadres transversaux) en une seule armature continue
    acier_tools = [(3, acier2)] + cadres_vols
    steel_fuse = occ.fuse([(3, acier1)], acier_tools, removeTool=True)
    aciers_dimtags = steel_fuse[0]

    # --- Noyau Coulé ---
    box_noyau = occ.addBox(epaisseur_U, epaisseur_U, 0, largeur - 2*epaisseur_U, hauteur - epaisseur_U, longueur)
    
    # On évide l'empreinte du squelette d'acier dans le noyau de béton 
    # (removeTool=False est crucial ici pour ne pas supprimer les aciers du modèle)
    noyau_cut = occ.cut([(3, box_noyau)], aciers_dimtags, removeTool=False)
    vol_Noyau = noyau_cut[0][0]

    # ==========================================
    # 3. Fragmentation pour maillage conforme
    # ==========================================
    # Fragmenter permet de forcer Gmsh à lier les noeuds aux interfaces communes
    entities_to_fragment = [vol_U, vol_Noyau] + aciers_dimtags
    out, out_map = occ.fragment([entities_to_fragment[0]], entities_to_fragment[1:])
    occ.synchronize()

    # Le dictionnaire `out_map` permet de retrouver les nouveaux tags générés par la fragmentation
    tags_U = [tag for dim, tag in out_map[0]]
    tags_Noyau = [tag for dim, tag in out_map[1]]
    
    tags_Aciers = []
    for i in range(2, len(out_map)):
        for dim, tag in out_map[i]:
            tags_Aciers.append(tag)
    
    # Nettoyage d'éventuels doublons
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
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)

    # Algorithme frontal 3D HXT : très performant pour les géométries 3D complexes (aciers)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10) 

    gmsh.model.mesh.generate(3)

    gmsh.write("poutre.msh")
    gmsh.fltk.run() # Décommentez pour lancer l'interface graphique et visualiser
    gmsh.finalize()
    
    return largeur, hauteur, epaisseur_U, longueur, es, phi

if __name__ == "__main__":
    create_geometry()