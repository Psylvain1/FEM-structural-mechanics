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
#from geometry_fibBeam_xaxis import *
import vtk
import matplotlib.pyplot as plt
  
#largeur, H, e_3DP, L,es, phi=create_geometry()
L=3.5
camera =[
        (L/2, 0,-0.1),  # Position de votre œil
        (L/2, 0, 0.1),  # Le point que vous regardiez
        (0.0, 1.0, 0.0)   # L'axe qui pointe vers le haut
    ]  

domain, cell_tags, facet_tags = io.gmshio.read_from_msh("poutre.msh", MPI.COMM_WORLD, 0, gdim=3)

mesh_preview(domain, cell_tags,camera)