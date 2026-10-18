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
from matplotlib.lines import Line2D
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
    # CORRECTION 1 : Le bras de levier de la semelle inférieure est (h/2) - (e_3DP/2)
    E1_Ainf1_d1 = (2 * E1 * e_3DP * (h / 2) * (h / 4)) + (E1 * e_3DP * (b - 2 * e_3DP) * ((h / 2) - e_3DP / 2))
    
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
    # CORRECTION 2 : L'acier est en bas, il tire yG vers le bas, donc on le soustrait.
    numerateur = EAy_sup - EAy_inf - EAy_acier
    denominateur = EA_sup + EA_inf + EA_acier
    
    yG = numerateur / denominateur
    
    # ==========================================
    # 5. MOMENT STATIQUE (ES)
    # ==========================================
    hauteur_bloc = np.linspace(0, h, 1000) 
    
    # Terme de base (Zone 1)
    terme_3DP = 2 * E1 * e_3DP * hauteur_bloc
    terme_remplissage = E2 * (b - 2 * e_3DP) * hauteur_bloc
    bras_de_levier = (h / 2 - yG) - hauteur_bloc / 2
    ES_base = (terme_3DP + terme_remplissage) * bras_de_levier

    # Terme lié à l'acier (Zone 2)
    terme_acier = Es * As * (yG + h / 2 - e_s)

    # CORRECTION 3 : Terme lié à la semelle inférieure (Zone 3)
    # On remplace E2 par E1 sur la largeur du noyau.
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
        ES_base - terme_acier + terme_b_inf_corr          # De h - e_3DP à h
    ]

    ES_max = np.select([zone1, zone2, zone3], valeurs, default=0)

    # ==========================================
    # 6. RIGIDITÉ FLEXIONNELLE TOTALE (EI_total)
    # ==========================================
    # (Votre code pour EI_total reste inchangé ici)
    
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

colors = cm_mpl.Blues(np.linspace(0.2, 0.8, 6))
for index,k in enumerate(np.linspace(0,0.05,6)):        
    # ==========================================
    # EXÉCUTION AVEC VOS VALEURS
    # ==========================================
    # Variables géométriques (en cm et cm²)
    h = 0.2 +k
    e_3DP = k #Varie entre 0 et b/2
    b = 0.1 + 2*k
    e_s = k + 0.01
    As = 0.000157  # Section pour 2 HA 10

    # Modules de Young (en MPa)
    E1 = 38000e6  # Béton 3D (38 GPa)
    E2 = 36000e6  # Béton remplissage (36 GPa)
    Es = 210000e6 # Acier (210 GPa)
    
    h = 0.26 + k
    e_3DP = k
    b = 0.16 + 2*k 
    e_s = 0.0325 + k 
    As = 0.0007037  
    E1 = 33000e6  
    E2 = 31000e6  
    Es = 210000e6

    # Appel de la fonction
    mon_yG,ES_max,EI_total,hauteur_bloc = calculer_yG(h, b, e_3DP, e_s, E1, E2, Es, As)

    def tau(V,ES,EI,b_y):
        return (V*ES)/(EI*b_y)
    
    Tau= tau(50000,ES_max,EI_total,b)*10**(-6)
    y_vals = (h/2 - hauteur_bloc)
    plt.plot(Tau,(h/2-hauteur_bloc),label= str(np.round(k*10**2,0)),c=colors[index])
    # --- AJOUT : Point rouge sur le max de Tau ---
    idx_max = np.argmax(Tau) # Trouve l'index de la valeur max de Tau
    plt.scatter(Tau[idx_max], y_vals[idx_max], color='red', zorder=5, s=20)
    # ---------------------------------------------
    # --- AJOUT : Point de l'interface et annotation ---
    # L'interface se trouve à la hauteur e_3DP par rapport au bas, soit y = -h/2 + k
    if k > 0: # On exclut k=0 car il n'y a pas d'interface (épaisseur nulle)
        y_interface = -h/2 + k
        idx_int = np.argmin(np.abs(y_vals - y_interface)) # Trouve l'index le plus proche de l'interface
        
        # Scatter de l'interface (par exemple en violet)
        plt.scatter(Tau[idx_int], y_vals[idx_int], color='#77DD77', edgecolor='black', linewidth=0.1, zorder=5, s=10)
        
       
    # --------------------------------------------------
    #plt.axvline(0.5, color='purple', linestyle='--', linewidth=0.8, alpha=0.7)  # Ligne horizontale pour l'interface




# ==========================================
# AJOUT DE LA DOUBLE FLÈCHE ET DU TEXTE
# ==========================================
# Coordonnée y pour placer la flèche (juste au-dessus de la limite inférieure de l'axe Y)
y_fleche = -h/2 + 0.01 

# Dessin de la double flèche entre 0.5 et 1.1 MPa
plt.annotate('', xy=(0.45, y_fleche), xytext=(1.15, y_fleche),
             arrowprops=dict(arrowstyle='<->', color='#77DD77', lw=1.2))

# Ajout du texte centré au-dessus de la flèche
# (0.8 est le milieu entre 0.5 et 1.1)
plt.text(0.8, y_fleche - 0.05, 'Rupture en \ncisaillement interface', 
         ha='center', va='bottom', fontsize=9, color='#77DD77')
plt.axvline(0.5, color='#77DD77', linestyle='--', linewidth=0.8, alpha=0.7,zorder=0)  # Ligne verticale pour l'interface
plt.axvline(1.1, color='#77DD77', linestyle='--', linewidth=0.8, alpha=0.7,zorder=0)  # Ligne verticale pour l'interface

# ==========================================

#plt.title("Cisaillement dans la section supérieure")
plt.ylabel(r"$y$ [m]")
plt.xlabel(r"$\tau$ [MPa]")
plt.grid()
# ==========================================
# GESTION DE LA LÉGENDE MULTIPLE
# ==========================================
# 1. On récupère les handles (lignes) et labels de la boucle for
handles_lignes, labels_lignes = plt.gca().get_legend_handles_labels()

# 2. On crée la PREMIÈRE légende (pour les courbes)
legende_courbes = plt.legend(handles=handles_lignes, labels=labels_lignes, 
                             title=r"Épaisseur coffrage $e_{3DP}$ [cm]:", loc='upper right')

# 3. On "sauvegarde" cette première légende sur le graphique
plt.gca().add_artist(legende_courbes)

# 4. On crée les "points" manuels pour la SECONDE légende
legend_max = Line2D([0], [0], marker='o', color='w', label=r'$\tau_{max}$ (Axe neutre)', markerfacecolor='red', markersize=6)
legend_interface = Line2D([0], [0], marker='o', color='w', label='Position de l\'interface', markerfacecolor='#77DD77', markersize=6)

# 5. On crée et affiche la SECONDE légende
plt.legend(handles=[legend_max, legend_interface], title="Points remarquables :", loc='lower right')
# ==========================================
plt.ylim(-h/2-0.002,h/2+0.002)
plt.xlim(-0.2,4)
plt.xticks(np.arange(0, 4.5, 1))
plt.savefig("Jourawski.pdf",bbox_inches="tight")
plt.show()