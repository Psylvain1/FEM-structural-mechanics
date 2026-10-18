import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
import pyvista as pv

def main():
    # ==========================================
    # 1. PARAMÈTRES DE L'ALGORITHME (MÉTHODE AM/SIMP PROGRESSIVE)
    # ==========================================
    L, H = 2.0, 0.5               
    nelx, nely = 80, 20           
    volfrac = 0.4                 
    
    # Paramètres de la continuation SIMP
    niternp = 20                  # Nombre d'itérations non-pénalisées (p=1)
    niter = 80                    # Nombre total d'itérations
    pmax = 4.0                    # Pénalité SIMP maximale
    exponent_update_freq = 4      # Fréquence de mise à jour de l'exposant
    thetamin = 0.001              # Densité minimale (vide)
    
    E0 = 1.0                      # Module de Young de la matière
    Emin = 1e-4                   
    nu = 0.3                      

    # ==========================================
    # 2. MAILLAGE ET ESPACES FENICSX
    # ==========================================
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], 
                                   [nelx, nely], mesh.CellType.quadrilateral)
    
    V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    V_rho = fem.functionspace(domain, ("DG", 0))

    rho = fem.Function(V_rho)
    rho.x.array[:] = volfrac

    # ==========================================
    # 3. FORMULATION MÉCANIQUE (UFL)
    # ==========================================
    p_c = fem.Constant(domain, default_scalar_type(1.0))

    def E(rho):
        return Emin + rho**p_c * (E0 - Emin)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def elastic_bilinear_form(u, v):
        return (1.0 / (1 + nu)) * (ufl.inner(epsilon(u), epsilon(v)) + 
                                   (nu / (1 - 2 * nu)) * ufl.tr(epsilon(u)) * ufl.tr(epsilon(v)))

    def strain_energy_density(uh):
        return (1.0 / (1 + nu)) * (ufl.inner(epsilon(uh), epsilon(uh)) + 
                                   (nu / (1 - 2 * nu)) * ufl.tr(epsilon(uh))**2)

    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    
    a = E(rho) * elastic_bilinear_form(u, v) * ufl.dx

    # ==========================================
    # 4. CONDITIONS AUX LIMITES ET CHARGEMENT
    # ==========================================
    def left_boundary(x): return np.isclose(x[0], 0.0)
    left_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left_boundary)
    left_dofs = fem.locate_dofs_topological(V_u, domain.topology.dim - 1, left_facets)
    bc = fem.dirichletbc(np.zeros(2, dtype=default_scalar_type), left_dofs, V_u)

    def right_boundary(x): return np.isclose(x[0], L)
    right_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right_boundary)
    facet_tags = mesh.meshtags(domain, domain.topology.dim - 1, right_facets, 
                               np.full_like(right_facets, 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    T = fem.Constant(domain, default_scalar_type((0.0, -1.0))) 
    L_form = ufl.inner(T, v) * ds(1)

    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # ==========================================
    # 5. BOUCLE D'OPTIMISATION TOPOLOGIQUE
    # ==========================================
    W_0 = fem.Function(V_rho) 
    history_rho, history_u, compliance_history = [], [], []
    
    old_compliance = 1e30
    exponent_counter = 0

    print("Début de l'optimisation...")
    for i in range(niter):
        uh = problem.solve()
        
        compliance_form = fem.form(ufl.inner(T, uh) * ds(1))
        compliance = fem.assemble_scalar(compliance_form)
        compliance_history.append(compliance)
        
        expr = fem.Expression(strain_energy_density(uh) * (E0 - Emin), 
                              V_rho.element.interpolation_points())
        W_0.interpolate(expr)
        W_array = W_0.x.array
        
        p_val = p_c.value
        rho_array = rho.x.array
        
        numerator = p_val * (rho_array**(2 * p_val)) * W_array
        
        l1, l2 = 0.0, 1e5
        while (l2 - l1) > 1e-6:
            lagrange = 0.5 * (l1 + l2)
            rho_new = np.clip((numerator / lagrange)**(1.0 / (p_val + 1.0)), thetamin, 1.0)
            if np.mean(rho_new) > volfrac: l1 = lagrange
            else: l2 = lagrange
                
        rho.x.array[:] = rho_new
        
        exponent_counter += 1
        if i >= niternp:
            if i == niternp:
                print("\n --- Début des itérations pénalisées ---")
                
            if (abs(compliance - old_compliance) < 0.01 * compliance_history[0]) and (exponent_counter > exponent_update_freq):
                gray_level = np.mean(4 * (rho.x.array - thetamin) * (1.0 - rho.x.array))
                new_p = min(p_c.value * (1.0 + 0.3**(1.0 + gray_level/2.0)), pmax)
                p_c.value = new_p
                exponent_counter = 0
                print(f"   >>> Exposant p mis à jour : {new_p:.3f}")

        old_compliance = compliance
        history_rho.append(rho.x.array.copy()) 
        history_u.append(uh.x.array.copy()) 
        
        print(f"Itération {i+1:02d}/{niter} | p = {p_c.value:.2f} | Volume = {np.mean(rho.x.array):.3f} | Compliance = {compliance:.3f}")

    # ==========================================
    # 6. VISUALISATION ET CRÉATION DU GIF
    # ==========================================
    print("\nGénération du GIF avec PyVista...")
    topology, cell_types, geometry = plot.vtk_mesh(V_u)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif("topopt_AM_continue.gif", fps=10)

    for iter_idx, (rho_data, u_data) in enumerate(zip(history_rho, history_u)):
        plotter.clear()
        grid.cell_data["Density"] = rho_data
        
        u_2d = u_data.reshape(-1, 2)
        u_3d = np.hstack([u_2d, np.zeros((u_2d.shape[0], 1))])
        grid.point_data["Displacement"] = u_3d
        
        max_disp = np.max(np.linalg.norm(u_3d, axis=1))
        warp_factor = (H * 0.5) / max_disp if max_disp > 1e-8 else 1.0
        warped_grid = grid.warp_by_vector("Displacement", factor=warp_factor)
        
        plotter.add_mesh(grid, color="white", opacity=0.0, show_edges=True, edge_color="lightgray", line_width=1)
        plotter.add_mesh(warped_grid, color="#e0f7fa", opacity=1.0, show_edges=True, edge_color="gray", line_width=1)
        plotter.add_mesh(warped_grid, scalars="Density", cmap="bone_r", clim=[0.0, 1.0], 
                         opacity="linear", show_edges=False, scalar_bar_args={"title": "Densite de matiere", "color": "black"})
        plotter.add_text(f"Iteration: {iter_idx + 1}/{niter}", position="upper_left", color="black", font_size=12)
        
        plotter.camera_position = 'xy'
        plotter.set_background("white")
        plotter.write_frame()

    plotter.close()
    print("Terminé ! Le fichier 'topopt_AM_continue.gif' a été créé.")

    # ==========================================
    # 7. EXPORT POUR IMPRESSION 3D (STL)
    # ==========================================
    print("\nGénération du fichier STL pour l'impression 3D...")
    
    # Épaisseur de l'extrusion (à adapter selon l'échelle de votre pièce)
    extrusion_thickness = 0.2 
    density_threshold = 0.5   # Conserve tout ce qui est plus dense que 50%
    
    # Applique la dernière densité calculée
    grid.cell_data["Density"] = history_rho[-1]
    
    # 1. Filtre les éléments pleins (crée un UnstructuredGrid)
    solid_grid = grid.threshold([density_threshold, 1.0], scalars="Density")
    
    # 2. Extrait la géométrie de surface (crée un PolyData)
    surface_2d = solid_grid.extract_geometry()
    
    # 3. Extrude suivant l'axe Z
    mesh_3d = surface_2d.extrude((0, 0, extrusion_thickness))
    
    # 4. Triangule (requis pour le format STL) et sauvegarde
    mesh_3d_tri = mesh_3d.triangulate()
    mesh_3d_tri.save("piece_optimisee.stl")
    
    print(f"Fichier 'piece_optimisee.stl' créé avec succès ! (Épaisseur : {extrusion_thickness})")

if __name__ == "__main__":
    main()