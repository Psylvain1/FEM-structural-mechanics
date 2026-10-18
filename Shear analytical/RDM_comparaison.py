import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_mpl
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import pyvista as pv
import os

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
    "figure.figsize": (width_in, height_in),
    "figure.dpi": 200,
    "font.family": "serif",
    "font.serif":  "CMU Serif", 
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 32,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 20, # Ajusté pour la légende du CSV
    "lines.linewidth": 2.0,
    "axes.linewidth": 0.1,
    "grid.linewidth": 0.5,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.2,
    "axes.unicode_minus": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0
})

def calculer_yG(h, b, e_3DP, e_s, E1, E2, Es, As):
    # ==========================================
    # 1. BÉTON SUPÉRIEUR
    # ==========================================
    EA_sup = (2 * E1 * e_3DP * (h / 2)) + (E2 * (b - 2 * e_3DP) * (h / 2))
    EAy_sup = EA_sup * (h / 4)
    
    # ==========================================
    # 2. BÉTON INFÉRIEUR
    # ==========================================
    EA_inf1 = (2 * E1 * e_3DP * (h / 2)) + (E1 * e_3DP * (b - 2 * e_3DP))
    E1_Ainf1_d1 = (2 * E1 * e_3DP * (h / 2) * (h / 4)) + (E1 * e_3DP * (b - 2 * e_3DP) * ((h / 2) - e_3DP/2))
    
    EA_inf2 = E2 * (b - 2 * e_3DP) * ((h / 2) - e_3DP)
    E2_Ainf2_d2 = EA_inf2 * (((h / 2) - e_3DP) / 2)
    
    EA_inf = EA_inf1 + EA_inf2
    EAy_inf = E1_Ainf1_d1 + E2_Ainf2_d2
    
    # ==========================================
    # 3. ACIER
    # ==========================================
    EA_acier = Es * As
    EAy_acier = EA_acier * ((h / 2) - e_s)
    
    # ==========================================
    # 4. CALCUL FINAL DE yG
    # ==========================================
    numerateur = EAy_sup - EAy_inf - EAy_acier
    denominateur = EA_sup + EA_inf + EA_acier
    yG = numerateur / denominateur
    
    # Moment statique
    hauteur_bloc = np.linspace(0, h, 1000) 
    terme_3DP = 2 * E1 * e_3DP * hauteur_bloc
    terme_remplissage = E2 * (b - 2 * e_3DP) * hauteur_bloc
    bras_de_levier = (h / 2 - yG) - hauteur_bloc / 2
    ES_base = (terme_3DP + terme_remplissage) * bras_de_levier

    terme_acier = Es * As * (yG + h / 2 - e_s)
    x_temp = hauteur_bloc - (h - e_3DP)
    delta_EA = (E1 - E2) * (b - 2 * e_3DP)
    
    bras_de_levier_corr = (h / 2 - yG) - ((h - e_3DP) + x_temp / 2)
    terme_b_inf_corr = delta_EA * x_temp * bras_de_levier_corr

    zone1 = hauteur_bloc <= (h - e_s)
    zone2 = (hauteur_bloc > (h - e_s)) & (hauteur_bloc <= (h - e_3DP))
    zone3 = hauteur_bloc > (h - e_3DP)

    valeurs = [
        ES_base,
        ES_base - terme_acier,
        ES_base - terme_acier + terme_b_inf_corr
    ]

    ES_max = np.select([zone1, zone2, zone3], valeurs, default=0)

    # ==========================================
    # 3. RIGIDITÉ FLEXIONNELLE TOTALE (EI_total)
    # ==========================================
    H_sup = (h / 2) - yG

    I_propre_sup_plein = (b * H_sup**3) / 12
    Trans_sup_plein = (b * H_sup) * (H_sup / 2)**2
    EI_sup_plein = E1 * (I_propre_sup_plein + Trans_sup_plein)
    
    I_propre_sup_creux = ((b - 2 * e_3DP) * H_sup**3) / 12
    Trans_sup_creux = ((b - 2 * e_3DP) * H_sup) * (H_sup / 2)**2
    EI_sup_creux = E1 * (I_propre_sup_creux + Trans_sup_creux)
    
    E1_I1_global_sup = EI_sup_plein - EI_sup_creux
    E2_I2_global_sup = E2 * (I_propre_sup_creux + Trans_sup_creux)
    EI_b_sup = E1_I1_global_sup + E2_I2_global_sup

    H_inf = (h / 2) + yG
    H_inf_inner = H_inf - e_3DP
    
    I_propre_inf_plein = (b * H_inf**3) / 12
    Trans_inf_plein = (b * H_inf) * (H_inf / 2)**2
    EI_inf_plein = E1 * (I_propre_inf_plein + Trans_inf_plein)
    
    I_propre_inf_creux = ((b - 2 * e_3DP) * H_inf_inner**3) / 12
    Trans_inf_creux = ((b - 2 * e_3DP) * H_inf_inner) * (H_inf_inner / 2)**2
    EI_inf_creux = E1 * (I_propre_inf_creux + Trans_inf_creux)
    
    E1_I1_global_inf = EI_inf_plein - EI_inf_creux
    E2_I2_global_inf = E2 * (I_propre_inf_creux + Trans_inf_creux)
    EI_b_inf = E1_I1_global_inf + E2_I2_global_inf

    dist_acier = (h / 2) + yG - e_s
    EI_acier = Es * (As * dist_acier**2)
    EI_total = EI_b_sup + EI_b_inf + EI_acier
    
    return yG, ES_max, EI_total, hauteur_bloc

def tau(V, ES, EI, b_y):
    return (V * ES) / (EI * b_y)

# Création de la figure 3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
colors = cm_mpl.Blues(np.linspace(0.6, 0.9, 1))

for index, k in enumerate(np.linspace(0.05, 0.05, 1)):        
    # ==========================================
    # EXÉCUTION AVEC VOS VALEURS
    # ==========================================
    h = 0.3 
    e_3DP = 0.04 
    b = 0.24
    e_s = 0.0525 
    As = 0.0007037  
    E1 = 33000e6  
    E2 = 31000e6  
    Es = 210000e6 

    mon_yG, ES_max, EI_total, hauteur_bloc = calculer_yG(h, b, e_3DP, e_s, E1, E2, Es, As)

    # ==========================================
    # CALCUL DU CISAILLEMENT
    # ==========================================
    V_effort = 15000
    Tau_moyen = tau(V_effort, ES_max, EI_total, b) * 10**(-6)

    zone_semelle = hauteur_bloc > (h - 0.02)
    Eb_haut = (2 * e_3DP * E1) + ((b - 2 * e_3DP) * E2)
    Eb_bas = b * E1
    Eb_y = np.where(zone_semelle, Eb_bas, Eb_haut)

    Tau_3DP = Tau_moyen * b * (E1 / Eb_y)
    Tau_cast = Tau_moyen * b * (E2 / Eb_y)
    Tau_cast[zone_semelle] = 0

    y_plot = h / 2 - hauteur_bloc

    # ==========================================
    # TRACÉS 3D (Théorie Jourawski)
    # ==========================================
    if e_3DP == 0:
        tau_bord = Tau_cast
    else:
        tau_bord = Tau_3DP

    ax.plot(tau_bord, np.full_like(y_plot, -b/2), y_plot, color=colors[index])
    ax.plot(tau_bord, np.full_like(y_plot, b/2), y_plot, color=colors[index])
    ax.plot(tau_bord, np.full_like(y_plot, -b/2 + e_3DP), y_plot, color=colors[index])
    ax.plot(tau_bord, np.full_like(y_plot, b/2-e_3DP), y_plot, color=colors[index])

    if e_3DP == 0:
        tau_centre = Tau_cast
    else:
        tau_centre = np.where(zone_semelle, Tau_3DP, Tau_cast)

    ax.plot(tau_centre, np.full_like(y_plot, 0), y_plot, color=colors[index], linestyle='--')

    idx_mid = np.argmin(np.abs(y_plot - 0))
    x_width = np.linspace(-b/2, b/2, 500)
    dans_paroi = (x_width <= -b/2 + e_3DP) | (x_width >= b/2 - e_3DP)
    
    if e_3DP == 0:
        tau_across = np.full_like(x_width, Tau_cast[idx_mid])
    else:
        tau_across = np.where(dans_paroi, Tau_3DP[idx_mid], Tau_cast[idx_mid])
        
    ax.plot(tau_across, -x_width, np.full_like(x_width, y_plot[idx_mid]), color=colors[index], linewidth=1.5,label='Théorie')
    
    y_out = [h/2, -h/2, -h/2, h/2, h/2]
    z_out = [-b/2, -b/2, b/2, b/2, -b/2]
    tau_0 = [0, 0, 0, 0, 0]
    ax.plot(tau_0, z_out, y_out, color="red", linewidth=1.0)

    if e_3DP > 0:
        y_in = [h/2, -h/2 + e_3DP, -h/2 + e_3DP, h/2]
        z_in = [-b/2 + e_3DP, -b/2 + e_3DP, b/2 - e_3DP, b/2 - e_3DP]
        tau_0_in = [0, 0, 0, 0]
        ax.plot(tau_0_in, z_in, y_in, color="red", linewidth=1.0)
    
    y_acier = -h/2 + e_s
    z_acier = [-0.01, 0.01]
    tau_0_acier = [0, 0]
    ax.plot(tau_0_acier, z_acier, [y_acier, y_acier], color="red", linewidth=1.5)
# ==========================================
# TRACÉ DU FICHIER CSV (FEniCS)
# ==========================================
csv_path = "results/tau_yz_line.csv" if os.path.exists("results/tau_yz_line.csv") else "tau_yz_line.csv"

if os.path.exists(csv_path):
    df_fenics = pd.read_csv(csv_path)
    y_fenics = df_fenics['y_shifted'].values
    tau_fenics = np.abs(df_fenics['tau_yz'].values) * 1e-6 
    
    ax.plot(tau_fenics, np.zeros_like(y_fenics), y_fenics, 
            color='orange', linewidth=2.5, linestyle='-', label='FEM')
    ax.legend(loc="center",bbox_to_anchor=(0.6, 0.85))
else:
    print(f"⚠️ Le fichier {csv_path} est introuvable. Le tracé FEniCSx a été ignoré.")

# ==========================================
# AJOUT LIGNE INTERFACE BÉTON
# ==========================================
y_line = -0.13
tau_vals = np.linspace(0, 1.51, 100)

# Ligne en pointillés
ax.plot(tau_vals, np.zeros_like(tau_vals), np.full_like(tau_vals, y_line), 
        color='red', linestyle='--', linewidth=2)

# Texte sous la ligne (zdir='x' pour l'orienter correctement dans la vue 3D projetée)
ax.text(0.8, 0, y_line - 0.005, "Interface entre le béton coulé\net le béton imprimé", 
        color='red', fontsize=16, ha='center', va='top', zdir='x')

# ==========================================
# FINALISATION DU GRAPHIQUE 3D/2D
# ==========================================
formatter_cm = ticker.FuncFormatter(lambda val, pos: f"{val * 100:g}")

ax.set_xlim(0, 1.51)
ax.set_xticks(np.arange(0, 1.6, 0.5))

ax.set_xlabel(r"$\tau$ [MPa]", labelpad=30)
ax.yaxis.set_major_formatter(formatter_cm)
ax.zaxis.set_major_formatter(formatter_cm)
ax.set_zlabel("y [cm]", labelpad=60)

ax.set_yticks([]) 
ax.tick_params(axis='z', pad=30)

ax.set_proj_type('ortho') 
ax.view_init(elev=0, azim=-90) 
ax.set_box_aspect((1, 1, 2))

plt.tight_layout()
plt.savefig("Jourawski_2D_diff.pdf", bbox_inches="tight", pad_inches=10)