import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_mpl
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager

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
# 1. Paramètres géométriques et matériaux fixes
# ==========================================
D = 0.26                 # Diamètre extérieur en mètres
R2 = D / 2.0             # Rayon extérieur (0.13 m)
epaisseur = 0.03         # Epaisseur du coffrage imprimé (3 cm)
R1 = R2 - epaisseur      # Rayon de l'interface (0.10 m)

nu = 0.2                 # Coefficient de Poisson du béton

# Module de Young de la coque (E2) fixe en Pascals 
E2 = 44e9  
# Module de cisaillement correspondant pour le calcul de torsion
G2 = E2 / (2 * (1 + nu)) 

# Moments quadratiques polaires de chaque partie (en m^4)
I0_1 = (np.pi * R1**4) / 2.0 
I0_2 = (np.pi * (R2**4 - R1**4)) / 2.0 

# ==========================================
# 2. Paramètres de calcul (Variables)
# ==========================================
# Moment de torsion appliqué CONSTANT (en kN.m)
Mt_kNm = 3 
Mt_Nm = Mt_kNm * 1000 

def calculer_limite_cisaillement(E_module_Pa, phi=0.12, d=0.0):
    # Cette fonction reçoit bien le Module de Young (E)
    E_eff_Pa = 0.75 * E_module_Pa
    E_eff_MPa = E_eff_Pa / 1e6
    f_ck = 10 * (E_eff_MPa / 19000) ** (1 / 0.38)
    f_ct = 0.2 * (f_ck ** (2/3))
    tau_0 = 1.08 * f_ct
    tau_final = tau_0 * (1 - d) * ((1 - phi) ** 2)
    return tau_final

# Liste des modules de Young du coeur (E1) en GPa
E1_list_GPa = np.linspace(44, 26, 7) 

# Vecteurs radiaux
r1 = np.linspace(0, R1, 100)    # Zone 1 : Cœur
r2 = np.linspace(R1, R2, 100)   # Zone 2 : Coque

# Génération du gradient de couleurs bleues
blue_colors = [cm_mpl.Blues(x) for x in np.linspace(0.4, 1.0, len(E1_list_GPa))]

# ==========================================
# PREMIER GRAPHIQUE
# ==========================================
plt.figure()

for i, E1_GPa in enumerate(E1_list_GPa):
    E1 = E1_GPa * 1e9
    # Calcul du module de cisaillement G1 pour les formules de mécanique
    G1 = E1 / (2 * (1 + nu))
    
    GJ_eq = G1 * I0_1 + G2 * I0_2
    
    tau1_Pa = (Mt_Nm * G1 / GJ_eq) * r1
    tau1_MPa = tau1_Pa / 1e6
    
    tau2_Pa = (Mt_Nm * G2 / GJ_eq) * r2
    tau2_MPa = tau2_Pa / 1e6
    
    c = blue_colors[i]
    
    # On affiche E1 dans la légende pour la clarté (les valeurs 44 à 26)
    plt.plot(r1, tau1_MPa, linewidth=2, color=c, label=f'$E_1$ = {E1_GPa:.0f} GPa')
    plt.plot(r2, tau2_MPa, linewidth=2, color=c)
    plt.plot([R1, R1], [tau1_MPa[-1], tau2_MPa[0]], linestyle=':', color=c, alpha=0.7)

plt.axhline(y=3.5, color='red', linestyle='-', linewidth=1.5, alpha=0.8,label="Limite en cisaillement du béton sain")

plt.axvline(x=R1, color='gray', linestyle='-.', linewidth=1.5, label=f'Interface ($R_1={R1:.2f}$ m)')
plt.axvline(x=R2, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Surface ext. ($R_2={R2:.2f}$ m)')

plt.xlabel("Position radiale $r$ [m]", fontsize=12)
plt.ylabel(r"$\tau$ [MPa]", fontsize=12)
plt.xlim(0, R2 * 1.05)
plt.ylim(0, None)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig("Evolution_cisaillement_torsion_E1_variable.pdf", bbox_inches="tight")
plt.show()

# ==========================================
# NOUVEAU GRAPHIQUE (Différence de module & Limites dynamiques)
# ==========================================
plt.figure()

E2_GPa = E2 / 1e9

# Vecteur des différences de module de Young (E2 - E1) de 0 à 18 GPa (E1 varie de 44 à 26)
delta_E_array = np.linspace(0, E2_GPa - 26, 18)

tau_jump_list = []
tau_max_coule_list = []
limite_d0_list = []
limite_d06_list = []

for delta_E in delta_E_array:
    # Retrouver E1 à partir de delta_E
    E1_pa = (E2_GPa - delta_E) * 1e9
    
    # Conversion en modules de cisaillement pour le calcul des contraintes
    G1_pa = E1_pa / (2 * (1 + nu))
    GJ_eq = G1_pa * I0_1 + G2 * I0_2
    
    # Cisaillement à l'interface (r = R1) calculé avec G
    tau1_R1 = (Mt_Nm * G1_pa / GJ_eq) * R1 / 1e6
    tau2_R1 = (Mt_Nm * G2 / GJ_eq) * R1 / 1e6
    
    # Calcul du saut de cisaillement et du cisaillement max dans la zone 1 (cœur)
    saut = abs(tau1_R1 - tau2_R1)
    tau_max_c = tau1_R1  
    
    tau_jump_list.append(saut)
    tau_max_coule_list.append(tau_max_c)
    
    # Calcul dynamique de la limite avec E1 (Module de Young)
    limite_d0_list.append(calculer_limite_cisaillement(E1_pa, d=0.0))
    limite_d06_list.append(calculer_limite_cisaillement(E1_pa, d=0.6))

# 1. Tracé du saut de cisaillement et max
colors = cm_mpl.Blues(np.linspace(0.6, 1.0, len(delta_E_array)))

plt.plot(delta_E_array, tau_jump_list,  marker='o', linestyle='-', color="red", label=r"$\Delta \tau_{max}$ (Saut à l'interface)")
plt.plot(delta_E_array, tau_max_coule_list, marker='s', markersize=4, linestyle='-.', color=colors[4], label=r"$\tau_{max,coulé}$")

# 2. Tracé des limites dynamiques (calculées via E)
plt.plot(delta_E_array, limite_d0_list, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=r"Limite int. intacte ($d=0$)")
plt.plot(delta_E_array, limite_d06_list, color='red', linestyle=':', linewidth=1.5, alpha=0.8, label=r"Limite int. dégradée ($d=0.6$)")

# 3. Ajout de la zone de dégradation dynamique
plt.fill_between(delta_E_array, limite_d06_list, limite_d0_list, color='red', alpha=0.15, label='Zone de dégradation ($0 \leq d \leq 0.6$)')

# Habillage du nouveau graphique
plt.xlabel(r"$\Delta E = E_2 - E_1$ [GPa]", fontsize=12)
plt.ylabel(r"$\tau$ [MPa]", fontsize=12)
plt.xlim(0, max(delta_E_array))
plt.xticks(np.linspace(0, max(delta_E_array), 7))

plt.ylim(0, 2.2) 
plt.yticks(np.linspace(0, 2, 5))

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.9)
#plt.title(r"Analyse des cisaillements critiques en fonction de $\Delta E$", fontsize=11, pad=10)

plt.tight_layout()
plt.savefig("Saut_et_Max_Cisaillement_vs_DeltaE.pdf", bbox_inches="tight")
plt.show()