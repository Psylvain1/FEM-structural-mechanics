import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_mpl
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import pyvista as pv

# --- 1. SETUP & STYLE ---
font_dir = '/usr/share/fonts/cmu'
if os.path.exists(font_dir):
    for font in os.listdir(font_dir):
        if font.endswith('.ttf'):
            font_manager.fontManager.addfont(os.path.join(font_dir, font))

cm = 1 / 2.54
width_in = 18.2 * cm
height_in = 12.0 * cm

# 2. Apply Matplotlib settings to match LaTeX
plt.rcParams.update({
    "figure.figsize": (width_in, height_in),
    "figure.dpi": 200,               
    "font.family": "serif",          
    "font.serif":  "CMU Serif", 
    "mathtext.fontset": "cm",        
    "font.size": 9,                  
    "axes.labelsize": 16,            
    "xtick.labelsize": 12,           
    "ytick.labelsize": 12,
    "legend.fontsize": 9,    
    "lines.linewidth": 2.0,          
    "axes.linewidth": 0.1,           
    "grid.linewidth": 0.5,           
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.5,
    "axes.unicode_minus": False,     
    "xtick.major.size": 0,  
    "ytick.major.size": 0,  
    "xtick.minor.size": 0,  
    "ytick.minor.size": 0
})

# ==========================================
# 1. Paramètres géométriques de la section
# ==========================================
D = 0.26                 # Diamètre extérieur en mètres
R = D / 2.0             # Rayon extérieur (0.13 m)
epaisseur = 0.03         # Epaisseur du coffrage imprimé (3 cm)
R1 = R - epaisseur  
# Moment quadratique polaire pour un cylindre plein (en m^4)
I0 = (np.pi * R**4) / 2.0 

# ==========================================
# 2. Paramètres de calcul
# ==========================================
# Vecteur du rayon r, allant du centre (0) au bord extérieur (R)
r = np.linspace(0, R, 200)

# Liste des moments de torsion appliqués
Mt_list_kNm = [1, 2, 3, 4, 5]

# Génération du gradient de couleurs bleues (de 0.4 à 1.0 pour éviter les bleus trop clairs)
blue_colors = [cm_mpl.Blues(x) for x in np.linspace(0.4, 1.0, len(Mt_list_kNm))]

# Initialisation de la figure
plt.figure()

# ==========================================
# 3. Calcul et Tracé des courbes
# ==========================================
for i, Mt_kNm in enumerate(Mt_list_kNm):
    # Conversion du moment en N.m
    Mt_Nm = Mt_kNm * 1000 
    
    # Calcul de la contrainte tau(r) en Pascals (N/m^2)
    tau_Pa = (Mt_Nm * r) / I0
    
    # Conversion des Pascals en MégaPascals (MPa)
    tau_MPa = tau_Pa / 1e6
    
    # Tracé de la courbe avec la couleur issue du gradient bleu
    plt.plot(r, tau_MPa, label=f'$M_t$ = {Mt_kNm} kN.m', linewidth=2, color=blue_colors[i])

# ==========================================
# 4. Ajout des seuils de contrainte (Nouveau)
# ==========================================
# Ajout de la surface rouge (alpha gère la transparence)
plt.axhspan(0.5, 1.5, color='red', alpha=0.15, label='Zone de rupture de l\'interface')

# Tracé des lignes horizontales
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
plt.axhline(y=1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
plt.axhline(y=3.5, color='red', linestyle='-', linewidth=1.5, alpha=0.8,label="Limite en cisaillement du béton sain")

# ==========================================
# 5. Habillage et mise en forme du graphique
# ==========================================
# Ligne verticale pour marquer la surface extérieure de la poutre
plt.axvline(x=R1, color='gray', linestyle='-.', linewidth=1.5, label=f'Interface ($R_1={R1:.2f}$ m)')
plt.axvline(x=R, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Surface extérieure ($R={R}$ m)')

# Textes et labels
#plt.title(f"Évolution de la contrainte de cisaillement en torsion\nCylindre de diamètre $D = {int(D*100)}$ cm", fontsize=14, pad=15)

plt.xlabel("Position radiale $r$ [m]", fontsize=12)
plt.ylabel(r"$\tau$ [MPa]", fontsize=12)

# Limites des axes pour un rendu propre
plt.xlim(0, R * 1.05)
plt.ylim(0, None)

# Grille et Légende
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, framealpha=0.9)

# Sauvegarde et affichage
plt.tight_layout()
plt.savefig("Evolution_cisaillement_torsion_seuils.pdf", bbox_inches="tight")
plt.show()