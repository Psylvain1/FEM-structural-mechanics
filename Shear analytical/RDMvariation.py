import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_mpl
import matplotlib.ticker as ticker
from matplotlib import font_manager
import pandas as pd
import os
import imageio.v2 as imageio # Required for creating the GIF

# --- 1. SETUP & STYLE ---
font_dir = '/usr/share/fonts/cmu'
if os.path.exists(font_dir):
    for font in os.listdir(font_dir):
        if font.endswith('.ttf'):
            font_manager.fontManager.addfont(os.path.join(font_dir, font))

cm = 1 / 2.54
width_in = 18.2 * cm
height_in = 12.0 * cm

plt.rcParams.update({
    "figure.figsize": (width_in, height_in),
    "figure.dpi": 200,
    "font.family": "serif",
    "font.serif":  "CMU Serif", 
    "mathtext.fontset": "cm", 
    "font.size": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,  
    "lines.linewidth": 1,
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
    
    #Moment statique
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

def calculer_limite_cisaillement(E_coule_Pa, phi=0.12, d=0.0):
    E_eff_Pa = 0.75 * E_coule_Pa
    E_eff_MPa = E_eff_Pa / 1e6
    f_ck = 10 * (E_eff_MPa / 19000) ** (1 / 0.38)
    f_ct = 0.2 * (f_ck ** (2/3))
    tau_0 = 1.08 * f_ct
    tau_final = tau_0 * (1 - d) * ((1 - phi) ** 2)
    return tau_final

# ==========================================
# CONSTANTS & SETUP FOR GIF
# ==========================================
k = 0.02
h = 0.28 + k
e_3DP = k 
b = 0.11 + 2*k
e_s = k + 0.01
As = 0.000157  
Es = 210000e6 
V_effort = 50000

E1_fixed = 44000e6  
E2_values = np.linspace(44000e6, 26000e6, 16) 

# Tracking variables for the final plot
delta_E_list = []
max_shear_diff_list = []
tau_limite_list = [] 
tau_limite_fissure_list = []
tau_cast_y12_list = [] # <-- Ajout liste pour la contrainte à y=-0.12
image_filenames = []

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = cm_mpl.Blues(np.linspace(0.6, 1.0, len(E2_values)))

for i, E2_val in enumerate(E2_values):
    ax.cla() 
    
    mon_yG, ES_max, EI_total, hauteur_bloc = calculer_yG(h, b, e_3DP, e_s, E1_fixed, E2_val, Es, As)
    Tau_moyen = tau(V_effort, ES_max, EI_total, b) * 10**(-6)

    zone_semelle = hauteur_bloc > (h - e_3DP)
    Eb_haut = (2 * e_3DP * E1_fixed) + ((b - 2 * e_3DP) * E2_val)
    Eb_bas = b * E1_fixed
    Eb_y = np.where(zone_semelle, Eb_bas, Eb_haut)

    Tau_3DP = Tau_moyen * b * (E1_fixed / Eb_y)
    Tau_cast = Tau_moyen * b * (E2_val / Eb_y)
    Tau_cast[zone_semelle] = 0
    y_plot = h / 2 - hauteur_bloc

    shear_diff = np.abs(Tau_3DP[~zone_semelle] - Tau_cast[~zone_semelle])
    max_diff = np.max(shear_diff)
    
    tau_limite_sain = calculer_limite_cisaillement(E2_val, d=0.0)
    tau_limite_fissure = calculer_limite_cisaillement(E2_val, d=0.6)
    
    delta_E = E1_fixed - E2_val
    delta_E_list.append(delta_E / 1e9) 
    max_shear_diff_list.append(max_diff)
    tau_limite_list.append(tau_limite_sain) 
    tau_limite_fissure_list.append(tau_limite_fissure) 

    tau_bord = Tau_3DP if e_3DP > 0 else Tau_cast
    tau_centre = np.where(zone_semelle, Tau_3DP, Tau_cast) if e_3DP > 0 else Tau_cast

    ax.plot(tau_bord,  np.full_like(y_plot, -b/2), y_plot, color=colors[i])
    ax.plot(tau_bord,  np.full_like(y_plot, b/2),y_plot, color=colors[i])
    ax.plot(tau_bord,  np.full_like(y_plot, -b/2 + e_3DP), y_plot, color=colors[i])
    ax.plot(tau_bord,  np.full_like(y_plot, b/2-e_3DP),y_plot, color=colors[i])
    ax.plot(tau_centre, np.full_like(y_plot, 0), y_plot, color=colors[i], linestyle='--')

    # Tracé transversal existant (à y=0)
    idx_mid = np.argmin(np.abs(y_plot - 0))
    x_width = np.linspace(-b/2, b/2, 500)
    dans_paroi = (x_width <= -b/2 + e_3DP) | (x_width >= b/2 - e_3DP)
    
    if e_3DP == 0:
        tau_across = np.full_like(x_width, Tau_cast[idx_mid])
    else:
        tau_across = np.where(dans_paroi, Tau_3DP[idx_mid], Tau_cast[idx_mid])
        
    ax.plot(tau_across, -x_width, np.full_like(x_width, y_plot[idx_mid]), color=colors[i], linewidth=1.5)
    
    # --- NOUVEAU : Tracé transversal à y = -0.12 ---
    idx_y12 = np.argmin(np.abs(y_plot - (-0.13)))
    if e_3DP == 0:
        tau_across_y12 = np.full_like(x_width, Tau_cast[idx_y12])
    else:
        tau_across_y12 = np.where(dans_paroi, Tau_3DP[idx_y12], Tau_cast[idx_y12])
        
    ax.plot(tau_across_y12, -x_width, np.full_like(x_width, y_plot[idx_y12]), color=colors[i], linewidth=1.5, linestyle='-.')
    
    # Enregistrement de la contrainte dans le béton coulé à y=-0.12
    tau_cast_y12_list.append(Tau_cast[idx_y12])
    # -----------------------------------------------

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

    ax.set_proj_type('ortho') 
    #ax.view_init(elev=25, azim=-30)
    ax.view_init(elev=0, azim=-90)  
    ax.set_box_aspect((0.8, 1, 1.5))
    
    ax.set_xlabel(r"$\tau$ [MPa]",labelpad=-5)
    ax.set_zlabel("y [cm]", labelpad=15)
    #ax.set_title(f"$E_1$ = 44 GPa | $E_2$ = {E2_val/1e9:.1f} GPa", fontsize=14)

    formatter_cm = ticker.FuncFormatter(lambda val, pos: f"{val * 100:g}")
    ax.yaxis.set_major_formatter(formatter_cm)
    ax.zaxis.set_major_formatter(formatter_cm)
    ax.set_yticks([]) 
    ax.tick_params(axis='z', pad=10)
    
    ax.set_zlim(-h/2, h/2)
    ax.set_zticks(np.arange(-h/2, h/2, 0.05))
    ax.tick_params(axis='x', pad=-5)
    ax.set_xlim(0, 2.5)    
    ax.set_xticks(np.arange(0, 2.6, 0.5))
    
    filename = f"frame_{i}.png"
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.5)
    image_filenames.append(filename)


# ==========================================
# COMPILE GIF
# ==========================================
with imageio.get_writer('Jourawski_Evolution.gif', mode='I', duration=0.5, loop=0) as writer:
    for filename in image_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in image_filenames:
    os.remove(filename)
print("Animation saved as Jourawski_Evolution.gif")





# ==========================================
# PLOT DIFFERENCE EVOLUTION & SHEAR LIMIT
# ==========================================
plt.figure()

# Courbe 1 : Différence de cisaillement
plt.plot(delta_E_list, max_shear_diff_list, marker='o', linestyle='-', color="red", label=r"$\Delta \tau_{max}$ (Saut à l'interface)")

# --- NOUVELLE COURBE 2 : Contrainte dans le béton coulé à y=-0.12 ---
plt.plot(delta_E_list, tau_cast_y12_list, marker='s', markersize=4, linestyle='-.', color=colors[4], label=r"$\tau_{coulé}$ (Fond de coffrage)")
# --------------------------------------------------------------------

# Courbe 3 : Limite de cisaillement (Matériau sain, d=0)
plt.plot(delta_E_list, tau_limite_list, linestyle='--', color='red', linewidth=1.5,  label=r"Limite int. intacte ($d=0$)")

# Courbe 4 : Limite de cisaillement (Matériau fissuré, d=0.6)
plt.plot(delta_E_list, tau_limite_fissure_list, linestyle=':', color='red', linewidth=1.0,label=r"Limite int. dégradée ($d=0.6$)")

# Surface rouge de la zone endommagée
plt.fill_between(delta_E_list, 
                 tau_limite_fissure_list, 
                 tau_limite_list, 
                 color='red', 
                 alpha=0.15,
                 label=r"Zone de dégradation ($0 \leq d \leq 0.6$)")

plt.xlabel(r"$\Delta E = E_1 - E_2$ [GPa]", fontsize=12)
plt.ylabel(r"Contrainte de cisaillement [MPa]", fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Shear_Difference_Evolution.pdf", bbox_inches="tight")
plt.show()