import gmsh


def create_geometry():
    gmsh.initialize()
    gmsh.model.add("Poutre_Composite_Armee")

    # ==========================================
    # Paramètres
    # ==========================================
    largeur = 0.2
    hauteur = 0.2
    epaisseur_U = 0.03
    longueur = 2.0

    es = epaisseur_U + 0.01  # Enrobage (hauteur du centre des aciers)
    phi = 0.01               # Diamètre des aciers
    R = phi / 2.0            # Rayon

    # Paramètre pour le Transfinite
    nb_noeuds_longueur = 21  # Cela créera 40 éléments (mailles) sur la longueur de 2m

    # ==========================================
    # 1. Géométrie 2D (OpenCASCADE)
    # ==========================================
    occ = gmsh.model.occ

    # Création du rectangle global et du noyau interne
    rect_global = occ.addRectangle(0, 0, 0, largeur, hauteur)
    rect_noyau = occ.addRectangle(epaisseur_U, epaisseur_U, 0, 
                                largeur - 2*epaisseur_U, hauteur - epaisseur_U)

    # Découpe booléenne pour obtenir le U parfait
    u_shape_cut = occ.cut([(2, rect_global)], [(2, rect_noyau)], removeTool=False)
    tag_U = u_shape_cut[0][0][1]

    # Création des deux disques d'acier 
    acier1 = occ.addDisk(largeur/2-2*phi, es, 0, R, R)
    acier2 = occ.addDisk(largeur/2 + 2*phi, es, 0, R, R)

    # On évide le noyau avec les aciers (Noyau - Aciers)
    noyau_cut = occ.cut([(2, rect_noyau)], [(2, acier1), (2, acier2)], removeTool=False)
    tag_Noyau = noyau_cut[0][0][1] 

    # Fusion conforme de toutes les surfaces
    occ.fragment([(2, tag_U)], [(2, tag_Noyau), (2, acier1), (2, acier2)])
    occ.synchronize()


    # 2. Extrusion 3D (Couches de Tétraèdres)
    # ==========================================
    # En enlevant recombine=True, on force Gmsh à diviser les prismes en tétraèdres
    extrusion_U = occ.extrude([(2, tag_U)], 0, 0, longueur, numElements=[20])
    extrusion_Noyau = occ.extrude([(2, tag_Noyau)], 0, 0, longueur, numElements=[20])
    extrusion_A1 = occ.extrude([(2, acier1)], 0, 0, longueur, numElements=[20])
    extrusion_A2 = occ.extrude([(2, acier2)], 0, 0, longueur, numElements=[20])

    vol_U = extrusion_U[1][1]
    vol_Noyau = extrusion_Noyau[1][1]
    vol_A1 = extrusion_A1[1][1]
    vol_A2 = extrusion_A2[1][1]

    occ.removeAllDuplicates()
    occ.synchronize()
    
    # ==========================================
    # 3. Groupes Physiques
    # ==========================================
    gmsh.model.addPhysicalGroup(3, [vol_U], name="Coffrage_U", tag=1)
    gmsh.model.addPhysicalGroup(3, [vol_Noyau], name="Coeur_Coule", tag=2)
    gmsh.model.addPhysicalGroup(3, [vol_A1, vol_A2], name="Aciers", tag=3)

   
    # ==========================================
    # 5. Paramètres du maillage et Génération
    # ==========================================
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)

    # Astuce optionnelle : L'algorithme de maillage frontal 3D (HXT) 
    # gère souvent mieux les maillages avec des "Transfinite curves" étirées.
    # gmsh.option.setNumber("Mesh.Algorithm3D", 10) 

    gmsh.model.mesh.generate(3)

    # Visualisation interactive
    #gmsh.fltk.run() 

    gmsh.write("poutre.msh")
    gmsh.finalize()
    return largeur, hauteur, epaisseur_U, longueur,es, phi

