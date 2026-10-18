import numpy as np
import matplotlib.pyplot as plt

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
import matplotlib.cm as cm_mpl
import matplotlib.ticker as ticker

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
    "font.size": 9,                  # Matches \footnotesize for ticks
    "axes.labelsize": 32,            # Matches \large for axis labels (x, y)
    "xtick.labelsize": 32,            # Ticks stay \footnotesize
    "ytick.labelsize": 32,
    "legend.fontsize": 32,    
    # --- Line & Grid Styling ---
    "lines.linewidth": 2.0,          # Matches line width= 2 pt
    "axes.linewidth": 0.1,           # Keeps the bounding box clean
    "grid.linewidth": 0.5,           # Matches grid=major
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.2,
    "axes.unicode_minus": False,     # Fixes minus signs in math text
    "xtick.major.size": 0,  # Removes X-axis dashes
    "ytick.major.size": 0,  # Removes Y-axis dashes
    "xtick.minor.size": 0,  # (Optional) Removes minor dashes if you have them
    "ytick.minor.size": 0
})


def calculer_yG(h, b, e_3DP, e_s, E1, E2, Es, As):
    """
    Calcule la position de l'axe neutre (yG) d'une section en U hétérogène.
    Toutes les unités de distance doivent être identiques (ex: cm).
    Toutes les unités de module doivent être identiques (ex: MPa).
    L'aire d'acier (As) doit être en unité de distance au carré (ex: cm²).
    """
    
    # ==========================================
    # 1. BÉTON SUPÉRIEUR
    # ==========================================
    # Calcul de la rigidité (EA) supérieure
    EA_sup = (2 * E1 * e_3DP * (h / 2)) + (E2 * (b - 2 * e_3DP) * (h / 2))
    
    # Calcul du moment statique (EAy) supérieur
    # Note : repris exactement selon votre formule (bras de levier = h/2)
    EAy_sup = EA_sup * (h / 4)
    
    # ==========================================
    # 2. BÉTON INFÉRIEUR
    # ==========================================
    # -- Partie 1 : Semelle et bras inférieurs en 3DP (E1) --
    #Vérifier l'équation en dessous
    EA_inf1 = (2 * E1 * e_3DP * (h / 2)) + (E1 * e_3DP * (b - 2 * e_3DP))
    E1_Ainf1_d1 = (2 * E1 * e_3DP * (h / 2) * (h / 4)) + (E1 * e_3DP * (b - 2 * e_3DP) * ((h / 2) - e_3DP/2))
    
    # -- Partie 2 : Remplissage inférieur (E2) --
    EA_inf2 = E2 * (b - 2 * e_3DP) * ((h / 2) - e_3DP)
    E2_Ainf2_d2 = EA_inf2 * (((h / 2) - e_3DP) / 2)
    
    # Bilan inférieur
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
    # Numérateur : Somme algébrique des moments statiques (avec vos signes)
    numerateur = EAy_sup - EAy_inf - EAy_acier
    
    # Dénominateur : Somme stricte des rigidités (tout est positif, c'est de la matière)
    denominateur = EA_sup + EA_inf + EA_acier
    
    # Calcul du centre de gravité
    yG = numerateur / denominateur
    
    # --- Affichage détaillé pour vérifier les étapes ---
    print(f"--- RÉSULTATS INTERMÉDIAIRES ---")
    print(f"Rigidité (EA) Supérieure : {EA_sup:,.0f}")
    print(f"Rigidité (EA) Inférieure : {EA_inf:,.0f}")
    print(f"Rigidité (EA) Acier      : {EA_acier:,.0f}")
    print(f"Rigidité Totale (Denom)  : {denominateur:,.0f}\n")
    
    print(f"Moment (EAy) Supérieur   : {EAy_sup:,.0f}")
    print(f"Moment (EAy) Inférieur   : {EAy_inf:,.0f}")
    print(f"Moment (EAy) Acier       : {EAy_acier:,.0f}")
    print(f"Moment Total (Num)       : {numerateur:,.0f}\n")
    
    print(f"=================================")
    print(f"Position de l'axe neutre (yG) : {yG:.3f}")
    print(f"=================================\n")

    #Moment statique
    hauteur_bloc = np.linspace(0, h, 1000) 
    
    # 2. Calcul du terme de base (ES_base) - Valable pour la Zone 1
    terme_3DP = 2 * E1 * e_3DP * hauteur_bloc
    terme_remplissage = E2 * (b - 2 * e_3DP) * hauteur_bloc
    bras_de_levier = (h / 2 - yG) - hauteur_bloc / 2

    ES_base = (terme_3DP + terme_remplissage) * bras_de_levier


    # 3. Calcul des termes à retrancher pour les Zones 2 et 3
    # Terme lié à l'acier (constant par rapport à y0)
    terme_acier = Es * As * (yG + h / 2 - e_s)

    # Terme lié au béton inférieur (dépend de y0, donc de hauteur_bloc)
    x_temp = hauteur_bloc - (h - e_3DP)
    delta_EA = (E1 - E2) * (b - 2 * e_3DP)
    
    # Le centre de gravité de la tranche x_temp est à y_mid = (h - e_3DP) + x_temp / 2
    # Le bras de levier est y_NA - y_mid
    bras_de_levier_corr = (h / 2 - yG) - ((h - e_3DP) + x_temp / 2)
    terme_b_inf_corr = delta_EA * x_temp * bras_de_levier_corr

    # Définition des 3 zones
    zone1 = hauteur_bloc <= (h - e_s)
    zone2 = (hauteur_bloc > (h - e_s)) & (hauteur_bloc <= (h - e_3DP))
    zone3 = hauteur_bloc > (h - e_3DP)

    valeurs = [
        ES_base,                                          # De 0 à h - e_s
        ES_base - terme_acier,                            # De h - e_s à h - e_3DP
        ES_base - terme_acier + terme_b_inf_corr         # De h - e_3DP à h
    ]

    # Calcul final du moment statique sur toute la section
    ES_max = np.select([zone1, zone2, zone3], valeurs, default=0)

    #print("Moment statique pour la partie au dessus de yg",ES_max)

    # ==========================================
    # 3. RIGIDITÉ FLEXIONNELLE TOTALE (EI_total)
    # ==========================================
    # --- BÉTON SUPÉRIEUR ---
    
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

    # --- BÉTON INFÉRIEUR ---
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

    # --- ACIER ---
    # Le bras de levier de l'acier est la distance exacte (sans division par 2)
    dist_acier = (h / 2) + yG - e_s
    EI_acier = Es * (As * dist_acier**2)

    # --- EI TOTAL ---
    EI_total = EI_b_sup + EI_b_inf + EI_acier

    
    return yG,ES_max, EI_total,hauteur_bloc

colors = cm_mpl.Blues(np.linspace(0.2, 0.8, 6))
for index,k in enumerate(np.linspace(0,0.05,6)):        
    # ==========================================
    # EXÉCUTION AVEC VOS VALEURS
    # ==========================================
    # Variables géométriques (en cm et cm²)
    h = 0.3 +k
    e_3DP = k #Varie entre 0 et b/2
    b = 0.1 + 2*k
    e_s = k + 0.01
    As = 0.000157  # Section pour 2 HA 10

    # Modules de Young (en MPa)
    E1 = 38000e6  # Béton 3D (38 GPa)
    E2 = 36000e6  # Béton remplissage (36 GPa)
    Es = 210000e6 # Acier (210 GPa)

    # Appel de la fonction
    mon_yG,ES_max,EI_total,hauteur_bloc = calculer_yG(h, b, e_3DP, e_s, E1, E2, Es, As)

    # --- NOUVELLE FONCTION DE CISAILLEMENT ---
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
        h = 0.3 + k
        e_3DP = k 
        b = 0.1 + 2*k
        e_s = k + 0.01
        As = 0.000157  

        E1 = 38000e6  
        E2 = 36000e6  
        Es = 210000e6 

        # Appel de la fonction pour la géométrie globale (à garder définie dans votre script)
        mon_yG, ES_max, EI_total, hauteur_bloc = calculer_yG(h, b, e_3DP, e_s, E1, E2, Es, As)

        # ==========================================
        # CALCUL DU CISAILLEMENT
        # ==========================================
        V_effort = 50000
        Tau_moyen = tau(V_effort, ES_max, EI_total, b) * 10**(-6)

        zone_semelle = hauteur_bloc > (h - e_3DP)
        Eb_haut = (2 * e_3DP * E1) + ((b - 2 * e_3DP) * E2)
        Eb_bas = b * E1
        Eb_y = np.where(zone_semelle, Eb_bas, Eb_haut)

        Tau_3DP = Tau_moyen * b * (E1 / Eb_y)
        Tau_cast = Tau_moyen * b * (E2 / Eb_y)
        Tau_cast[zone_semelle] = 0

        y_plot = h / 2 - hauteur_bloc

        # ==========================================
        # TRACÉS 3D POUR CETTE ITÉRATION (Axes modifiés)
        # Nouveaux axes : X = Contrainte, Y = Hauteur, Z = -Largeur
        # ==========================================
        
        # Choix de la contrainte sur les bords externes selon la présence du coffrage
        if e_3DP == 0:
            tau_bord = Tau_cast
        else:
            tau_bord = Tau_3DP

        # 1. Profil vertical sur le bord externe (Anciennement X = b/2 et -b/2)
        # Note : le -b/2 devient b/2 et inversement à cause du "-Z" demandé
        ax.plot(tau_bord,  np.full_like(y_plot, -b/2), y_plot,
                color=colors[index])
        ax.plot(tau_bord,  np.full_like(y_plot, b/2),y_plot, color=colors[index])

        ax.plot(tau_bord,  np.full_like(y_plot, -b/2 + e_3DP), y_plot,
                color=colors[index])
        ax.plot(tau_bord,  np.full_like(y_plot, b/2-e_3DP),y_plot, color=colors[index])

        # Choix de la contrainte au centre de la section
        if e_3DP == 0:
            tau_centre = Tau_cast
        else:
            tau_centre = np.where(zone_semelle, Tau_3DP, Tau_cast)

        # 2. Profil vertical au centre de la poutre
        # En pointillés pour le différencier visuellement
        ax.plot(tau_centre,  np.full_like(y_plot, 0), y_plot,
                color=colors[index], linestyle='--')

        # 3. Profil transversal (horizontal) à mi-hauteur (y=0)
        idx_mid = np.argmin(np.abs(y_plot - 0))
        x_width = np.linspace(-b/2, b/2, 500)
        
        dans_paroi = (x_width <= -b/2 + e_3DP) | (x_width >= b/2 - e_3DP)
        
        if e_3DP == 0:
            tau_across = np.full_like(x_width, Tau_cast[idx_mid])
        else:
            tau_across = np.where(dans_paroi, Tau_3DP[idx_mid], Tau_cast[idx_mid])
            
        # Tracé de la ligne transversale
        # On passe tau_across en X, hauteur en Y, et l'opposé de la largeur en Z (-x_width)
        ax.plot(tau_across, -x_width, np.full_like(x_width, y_plot[idx_mid]), 
                color=colors[index], linewidth=1.5)
        
        # Contour extérieur de la poutre
        y_out = [h/2, -h/2, -h/2, h/2, h/2]
        z_out = [-b/2, -b/2, b/2, b/2, -b/2]
        tau_0 = [0, 0, 0, 0, 0]
        ax.plot(tau_0, z_out, y_out, color="red", linewidth=1.0)

        # Contour intérieur (l'interface en U si présence d'un coffrage)
        if e_3DP > 0:
            # Ligne de gauche, ligne du fond, ligne de droite
            y_in = [h/2, -h/2 + e_3DP, -h/2 + e_3DP, h/2]
            z_in = [-b/2 + e_3DP, -b/2 + e_3DP, b/2 - e_3DP, b/2 - e_3DP]
            tau_0_in = [0, 0, 0, 0]
            ax.plot(tau_0_in, z_in, y_in, color="red", linewidth=1.0)
        
        # Tracé des armatures (acier)
        y_acier = -h/2 + e_s
        z_acier = [-0.01, 0.01] # Largeur totale de 0.02 centrée sur 0
        tau_0_acier = [0, 0]
        # J'utilise du noir et un trait un peu plus épais pour bien les distinguer
        ax.plot(tau_0_acier, z_acier, [y_acier, y_acier], color="red", linewidth=1.5)

"""
# --- FINALISATION DU GRAPHIQUE 3D ---
ax.set_xlabel(r"$\tau$ [MPa]", labelpad=20)

formatter_cm = ticker.FuncFormatter(lambda val, pos: f"{val * 100:g}")

# Application du formatage sur l'axe Y (Largeur) et l'axe Z (Hauteur)
ax.yaxis.set_major_formatter(formatter_cm)
ax.zaxis.set_major_formatter(formatter_cm)

ax.set_ylabel("z [cm] ", labelpad=20)
ax.set_zlabel("y [cm]", labelpad=20)
#ax.set_title("Cartographie 3D du cisaillement dans la section")

# Orientation initiale de la caméra (Elevation, Azimuth) pour bien voir le relief
ax.view_init(elev=20, azim=-40)
ax.set_box_aspect((1, 1, 2))
#ax.view_init(elev=+25, azim=0,roll=90)




plt.tight_layout()
plt.savefig("Jourawski_3D.pdf", bbox_inches="tight")

"""

ax.set_proj_type('ortho') 

# elev=0 removes the vertical tilt, azim=-90 looks directly down the Y-axis (your z [cm])
ax.view_init(elev=0, azim=-90) 
ax.set_box_aspect((1, 1, 2))


# --- FINALISATION DU GRAPHIQUE 3D ---
ax.set_xlabel(r"$\tau$ [MPa]", labelpad=30)

formatter_cm = ticker.FuncFormatter(lambda val, pos: f"{val * 100:g}")

# Application du formatage sur l'axe Y (Largeur) et l'axe Z (Hauteur)
ax.yaxis.set_major_formatter(formatter_cm)
ax.zaxis.set_major_formatter(formatter_cm)


ax.set_zlabel("y [cm]", labelpad=60)

# --- NOUVEAUX PARAMÈTRES POUR LES GRADUATIONS ---
# Masquer les graduations de l'axe Y de matplotlib (votre largeur z)
ax.set_yticks([]) 

# Décaler les graduations de l'axe Z de matplotlib (votre hauteur y) de -10 pt
ax.tick_params(axis='z', pad=30)

# --- VUE 2D ORTHOGRAPHIQUE ---
ax.set_proj_type('ortho') 
ax.view_init(elev=30, azim=-30) 
ax.set_box_aspect((1, 1, 2))

plt.tight_layout()
# Ajoutez pad_inches pour forcer une marge autour de la figure (en pouces, ex: 0.5)
plt.savefig("Jourawski_2D_diff.pdf", bbox_inches="tight", pad_inches=0.5)