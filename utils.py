import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from utils import *
import pyvista
from dolfinx import plot
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

import numpy.typing as npt
import dolfinx
from dolfinx import fem
import ufl
import basix
import typing
from petsc4py import PETSc

from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.bcs import DirichletBC


class BlockedLinearProblem:
    """Class for solving a blocked linear variational problem."""

    def __init__(
        self,
        a_blocked_compiled,
        L_blocked_compiled,
        w: list[dolfinx.fem.Function],
        bcs: list[DirichletBC] = [],
        petsc_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for a blocked linear variational problem.

        Args:
            a: A bilinear UFL form, the left hand side of the
                variational problem.
            L: A linear UFL form, the right hand side of the variational
                problem.
            w: A list of Functions to store the result.
            bcs: A list of Dirichlet boundary conditions.

        """

        self._a = a_blocked_compiled
        self._L = L_blocked_compiled
        self._A = fem.petsc.create_matrix_block(self._a)
        self._b = fem.petsc.create_vector_block(self._L)
        self.w = w

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self._b.getComm().tompi4py())
        self._solver.setOperators(self._A)
        self._solver.setFromOptions()

        # # Set matrix and vector PETSc options
        self._A.setFromOptions()
        self._b.setFromOptions()

        self._x = self._A.createVecRight()
        self.bcs = bcs

    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()

    def assemble(self):
        # Assemble lhs
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_block(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(self._b, self._L, self._a, bcs=self.bcs)

    def solve(self) -> _Function:
        """Solve the problem."""
        self.assemble()

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        # self._x.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )
        assert self._solver.getConvergedReason() > 0, "Solve failed"

        blocked_maps = [
            (
                si.function_space.dofmap.index_map,
                si.function_space.dofmap.index_map_bs,
            )
            for si in self.w
        ]
        local_values = dolfinx.cpp.la.petsc.get_local_vectors(self._x, blocked_maps)
        for lx, u in zip(local_values, self.w):
            u.x.array[:] = lx
            # u.x.scatter_forward()

        return self.w

    @property
    def L(self) -> fem.Form:
        """The compiled linear form"""
        return self._L

    @property
    def a(self) -> fem.Form:
        """The compiled bilinear form"""
        return self._a

    @property
    def A(self) -> PETSc.Mat:
        """Matrix operator"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Right-hand side vector"""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """Linear solver object"""
        return self._solver


def transfer_meshtags_to_submesh(
    mesh: dolfinx.mesh.Mesh,
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    sub_vertex_to_parent: npt.NDArray[np.int32],
    sub_cell_to_parent: npt.NDArray[np.int32],
) -> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.

    Args:
        mesh: Mesh containing the meshtags
        entity_tag: The meshtags object to transfer
        submesh: The submesh to transfer the `entity_tag` to
        sub_to_vertex_map: Map from each vertex in `submesh` to the corresponding
            vertex in the `mesh`
        sub_cell_to_parent: Map from each cell in the `submesh` to the corresponding
            entity in the `mesh`
    Returns:
        The entity tag defined on the submesh, and a map from the entities in the
        `submesh` to the entities in the `mesh`.

    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(entity_tag.dim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet

    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


def create_piecewise_constant_field(
    domain, cell_markers, property_dict, name=None, default_value=0
):
    """Create a piecewise constant field with different values per subdomain.

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    cell_markers : MeshTag
        cell marker MeshTag
    property_dict : dict
        A dictionary mapping region tags to physical values {tag: value}

    Returns
    -------
    A DG-0 function
    """
    V0 = dolfinx.fem.functionspace(domain, ("DG", 0))
    k = dolfinx.fem.Function(V0, name=name)
    k.x.array[:] = default_value
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k


def interpolate_submesh_to_parent(u_parent, u_sub, sub_to_parent_cells):
    """
    Interpolate results from functions defined in submeshes to a function defined on the parent mesh.

    Parameters
    ----------
    u_parent : dolfinx.fem.Function
        Parent function to interpolate to.
    u_sub : list[dolfinx.fem.Function]
        _description_
    sub_to_parent_cells : list
        Submesh cells to parent cells mapping
    """
    V_parent = u_parent.function_space
    V_sub = u_sub.function_space
    for i, cell in enumerate(sub_to_parent_cells):
        bs = V_parent.dofmap.bs
        bs_sub = V_sub.dofmap.bs
        assert bs == bs_sub
        parent_dofs = V_parent.dofmap.cell_dofs(cell)
        sub_dofs = V_sub.dofmap.cell_dofs(i)
        for p_dof, s_dof in zip(parent_dofs, sub_dofs):
            for j in range(bs):
                u_parent.x.array[p_dof * bs + j] = u_sub.x.array[s_dof * bs + j]


def interface_int_entities(
    msh,
    interface_facets,
    domain_to_domain_0,
    domain_to_domain_1,
):
    """
    This function computes the integration entities (as a list of pairs of
    (cell, local facet index) pairs) required to assemble mixed domain forms
    over the interface. It assumes there is a domain with two sub-domains,
    domain_0 and domain_1, that have a common interface. Cells in domain_0
    correspond to the "+" restriction and cells in domain_1 correspond to
    the "-" restriction.

    Parameters:
        interface_facets: A list of facets on the interface
        domain_0_cells: A list of cells in domain_0
        domain_1_cells: A list of cells in domain_1
        c_to_f: The cell to facet connectivity for the domain mesh
        f_to_c: the facet to cell connectivity for the domain mesh
        facet_imap: The facet index_map for the domain mesh
        domain_to_domain_0: A map from cells in domain to cells in domain_0
        domain_to_domain_1: A map from cells in domain to cells in domain_1

    Returns:
        A tuple containing:
            1) The integration entities
            2) A modified map (see HACK below)
            3) A modified map (see HACK below)
    """
    # Create measure for integration. Assign the first (cell, local facet)
    # pair to the cell in domain_0, corresponding to the "+" restriction.
    # Assign the second pair to the domain_1 cell, corresponding to the "-"
    # restriction.
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    # FIXME This can be done more efficiently
    interface_entities = []
    domain_to_domain_0_new = np.array(domain_to_domain_0)
    domain_to_domain_1_new = np.array(domain_to_domain_1)
    for facet in interface_facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            cells = f_to_c.links(facet)
            assert len(cells) == 2
            if domain_to_domain_0[cells[0]] > 0:
                cell_plus = cells[0]
                cell_minus = cells[1]
            else:
                cell_plus = cells[1]
                cell_minus = cells[0]
            assert (
                domain_to_domain_0[cell_plus] >= 0
                and domain_to_domain_0[cell_minus] < 0
            )
            assert (
                domain_to_domain_1[cell_minus] >= 0
                and domain_to_domain_1[cell_plus] < 0
            )

            local_facet_plus = np.where(c_to_f.links(cell_plus) == facet)[0][0]
            local_facet_minus = np.where(c_to_f.links(cell_minus) == facet)[0][0]

            interface_entities.extend(
                [cell_plus, local_facet_plus, cell_minus, local_facet_minus]
            )

            # FIXME HACK cell_minus does not exist in the left submesh, so it
            # will be mapped to index -1. This is problematic for the
            # assembler, which assumes it is possible to get the full macro
            # dofmap for the trial and test functions, despite the restriction
            # meaning we don't need the non-existant dofs. To fix this, we just
            # map cell_minus to the cell corresponding to cell plus. This will
            # just add zeros to the assembled system, since there are no
            # u("-") terms. Could map this to any cell in the submesh, but
            # I think using the cell on the other side of the facet means a
            # facet space coefficient could be used
            domain_to_domain_0_new[cell_minus] = domain_to_domain_0[cell_plus]
            # Same hack for the right submesh
            domain_to_domain_1_new[cell_plus] = domain_to_domain_1[cell_minus]

    return interface_entities, domain_to_domain_0_new, domain_to_domain_1_new



def create_piecewise_constant_field(
    domain, cell_markers, property_dict, name=None, default_value=0
):
    """Create a piecewise constant field with different values per subdomain.

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    cell_markers : MeshTag
        cell marker MeshTag
    property_dict : dict
        A dictionary mapping region tags to physical values {tag: value}

    Returns
    -------
    A DG-0 function
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    k.x.array[:] = default_value
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k

import pyvista as pv
from dolfinx import plot

def mesh_preview(domain, cell_tags,camera):
    """
    Génère un aperçu 3D interactif du maillage composite 
    en séparant le coffrage, le cœur et les aciers.
    """
    pv.global_theme.font.family = 'times'  # Assurez-vous que la police est installée sur votre système
    pv.global_theme.font.size = 14 # Ajuste la taille selon tes besoins
    pv.global_theme.font.title_size = 16
    pv.global_theme.font.label_size = 14

    # Initialiser le traceur PyVista
    plotter = pv.Plotter(window_size=[2600, 2600])
    
    # Extraire la topologie 3D du domaine FEniCSx pour PyVista
    topology, cell_types, x = plot.vtk_mesh(domain, domain.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    
    # Associer les tags de matériaux Gmsh aux cellules du maillage
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    grid.cell_data["Materiau"] = cell_tags.values[:num_cells_local]
    
    # Isoler et afficher le Coffrage U (Tag 1)
    try:
        mesh_u = grid.threshold([1, 1], scalars="Materiau")
        if mesh_u.n_cells > 0:
            # 1. Les faces transparentes (sans arêtes)
            plotter.add_mesh(mesh_u, color="#e67e22", opacity=0.25, 
                             show_edges=False, label="Coffrage en béton imprimé")
            
            # 2. Les arêtes noires 100% opaques par-dessus
            edges_u = mesh_u.extract_all_edges()
            plotter.add_mesh(edges_u, color="red", opacity=0.2, line_width=0.5)
    
    except ValueError:
        pass

    # Isoler et afficher le Cœur Coulé (Tag 2)
    try:
        mesh_noyau = grid.threshold([2, 2], scalars="Materiau")
        if mesh_noyau.n_cells > 0:
            # 1. Les faces semi-transparentes
            plotter.add_mesh(mesh_noyau, color="#3498db", opacity=0.5, 
                             show_edges=False, label="Béton Coulé")
            
            # 2. Les arêtes noires 100% opaques
            edges_noyau = mesh_noyau.extract_all_edges()
            plotter.add_mesh(edges_noyau, color="red", opacity=0.2, line_width=0.5)
    except ValueError:
        pass

    # Isoler et afficher les Aciers (Tag 3)
    try:
        mesh_aciers = grid.threshold([3, 3], scalars="Materiau")
        if mesh_aciers.n_cells > 0:
            plotter.add_mesh(mesh_aciers, color="#2c3e50", opacity=1.0, 
                             show_edges=True, edge_color="black", label="Aciers")
    except ValueError:
        pass

    # Configuration de la scène
    plotter.add_legend(bcolor=None, face="circle",size=(0.25, 0.25))
    #plotter.view_isometric()
    
    plotter.camera_position = camera
  
    """
    [
        (0.8, 0.8, 4),  # Position de votre œil
        (0.1, 0.1, 1.0),  # Le point que vous regardiez
        (0.0, 1.0, 0.0)   # L'axe qui pointe vers le haut
    ]
    """
    plotter.show_axes()

    # 1. Récupérer les limites du maillage
    bounds = grid.bounds 
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # 2. Calculer les dimensions réelles
    dim_x = xmax - xmin
    dim_y = ymax - ymin
    dim_z = zmax - zmin
    
    # 3. Définir les points d'ancrage en 3D (au milieu des arêtes inférieures/latérales)
    points_ancrage = [
        [(xmin + xmax) / 2, ymin, zmin],  # Milieu de l'arête X
        [xmin, (ymin + ymax) / 2, zmin],  # Milieu de l'arête Y (Longueur)
        [xmin, ymin, (zmin + zmax) / 2]   # Milieu de l'arête Z (Hauteur)
    ]
    
    # 4. Créer les textes correspondants
    labels_dimensions = [
        f"Longueur: {dim_x:.2f} m", 
        f"Hauteur: {dim_y:.2f} m", 
        f"Largeur: {dim_z:.2f} m"
    ]
    
    # 5. Ajouter les étiquettes dans la scène 3D
    plotter.add_point_labels(
        points_ancrage,
        labels_dimensions,
        italic=True,
        font_size=32,
        text_color="black",
        point_color="red",      # Couleur du point sur l'arête
        point_size=8,           # Taille du point (mettre 0 si tu veux le cacher)
        shape=None,             # Mettre 'rect' si tu veux un fond derrière le texte
        always_visible=True     # Force le texte à s'afficher par-dessus le maillage
    )
    
    
    plotter.show()
    # 1. Save the standard screenshot FIRST
    plotter.screenshot("results/3Dmodel.png", scale=3)
    

    


# ==========================================
# 3. Tracé des lois de Traction-Séparation (CZM)
# ==========================================

def plot_czm_laws(Gc_n,sig_max_n,K_pen,Gc_t,sig_max_t,K_pent):
    # Paramètres Interface (Valeurs tirées de votre code FEniCSx)

    # Calcul des points critiques - Mode I (Normal)
    delta_0_n = sig_max_n / K_pen
    delta_f_n = 2.0 * Gc_n / sig_max_n

    # Calcul des points critiques - Mode II (Tangentiel)
    delta_0_t = sig_max_t / K_pent
    delta_f_t = 2.0 * Gc_t / sig_max_t

    # --- Construction des vecteurs pour le Mode I ---
    # On commence à -2% de l'ouverture de rupture pour visualiser la forte pente en compression
    d_n = np.array([-0.02 * delta_f_n, 0, delta_0_n, delta_f_n, delta_f_n * 1.2])
    
    # Contraintes associées (Loi de Hooke en compression, puis Bilinéaire en traction)
    T_n = np.array([-0.02 * delta_f_n * K_pen, 0, sig_max_n, 0, 0])

    # --- Construction des vecteurs pour le Mode II ---
    # Le cisaillement est symétrique, on trace juste l'ouverture positive
    d_t = np.array([0, delta_0_t, delta_f_t, delta_f_t * 1.2])
    T_t = np.array([0, sig_max_t, 0, 0])

    # Conversion des unités pour l'affichage (Ouverture en mm, Contrainte en MPa)
    d_n_mm, T_n_MPa = d_n * 1e3, T_n / 1e6
    d_t_mm, T_t_MPa = d_t * 1e3, T_t / 1e6

    # --- Création des graphiques ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1 : Mode I (Traction et Compression)
    # Ligne bleue : Comportement en traction (rupture bilinéaire)
    ax1.plot(d_n_mm[1:], T_n_MPa[1:], 'b-o', linewidth=2, label='Traction (Adoucissement)')
    # Ligne rouge : Comportement en compression (Contact rigide / Pénalité)
    ax1.plot(d_n_mm[:2], T_n_MPa[:2], 'r-', linewidth=3, label='Compression (Pénalité infinie)')
    
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_title('Mode I : Normal (Traction / Compression)')
    ax1.set_xlabel('Ouverture normale $\Delta_n$ (mm)')
    ax1.set_ylabel('Contrainte de cohésion $T_n$ (MPa)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # On limite l'axe Y vers le bas pour ne pas écraser le graphique (car K_pen est immense)
    ax1.set_ylim(bottom=-sig_max_n/1e6 * 2.5)
    ax1.legend()

    # Graphique 2 : Mode II (Cisaillement)
    ax2.plot(d_t_mm, T_t_MPa, 'g-s', linewidth=2, label='Cisaillement')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_title('Mode II : Tangentiel (Cisaillement)')
    ax2.set_xlabel('Glissement tangentiel $\Delta_t$ (mm)')
    ax2.set_ylabel('Contrainte de cisaillement $T_t$ (MPa)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import os
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import pyvista as pv

def graphics_setup():
    # --- 1. SETUP & STYLE ---
    font_dir = '/usr/share/fonts/cmu'
    if os.path.exists(font_dir):
        for font in os.listdir(font_dir):
            if font.endswith('.ttf'):
                font_manager.fontManager.addfont(os.path.join(font_dir, font))


    cm = 1 / 2.54
    width_in = 14.2 * cm
    height_in = 12.0 * cm

    # 2. Apply Matplotlib settings to match LaTeX
    plt.rcParams.update({
        # --- Figure Size & Resolution ---
        "figure.figsize": (width_in, height_in),
        "figure.dpi": 200,               # Matches convert={density=600}
        
        # --- Font / Police de caractères ---
        "font.family": "serif",          # Forces serif fonts
        "font.serif":  "CMU Serif", 
        "mathtext.fontset": "cm",        # Matches LaTeX math rendering for $\xi_2$
        
        # --- Font Sizes ---
        "font.size": 16,                  # Matches \footnotesize for ticks
        "axes.labelsize": 16,            # Matches \large for axis labels (x, y)
        "xtick.labelsize": 12,            # Ticks stay \footnotesize
        "ytick.labelsize": 12,
        "legend.fontsize": 9,    
        # --- Line & Grid Styling ---
        "lines.linewidth": 2.0,          # Matches line width= 2 pt
        "axes.linewidth": 0.1,           # Keeps the bounding box clean
        "grid.linewidth": 0.5,           # Matches grid=major
        "grid.color": "#b0b0b0",
        "grid.alpha": 0.5,
        "axes.unicode_minus": False,     # Fixes minus signs in math text
        "xtick.major.size": 0,  # Removes X-axis dashes
        "ytick.major.size": 0,  # Removes Y-axis dashes
        "xtick.minor.size": 0,  # (Optional) Removes minor dashes if you have them
        "ytick.minor.size": 0
    })
    