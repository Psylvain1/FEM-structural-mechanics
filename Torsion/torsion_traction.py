import numpy as np
import matplotlib.pyplot as plt



# ==========================================
# Paramètres globaux (Section de la poutre)
# ==========================================
R = 20.0             # Rayon extérieur de la poutre
R2 = 16.0            # Rayon intérieur (Béton coulé)
center_y = -R / 2.5  # Position Y du prisme
center_z = R / 2.5   # Position Z du prisme

# ==========================================
# Paramètres locaux (Le prisme)
# ==========================================
theta_deg = 45
theta = np.radians(theta_deg)
dA = 10.0            # Taille réduite du prisme pour rentrer dans le schéma

# Dimensions géométriques du prisme
H = dA * np.cos(theta) 
W = dA * np.sin(theta) 

# Coordonnées du coin inférieur gauche du prisme pour que son centre 
# géométrique tombe pile sur (center_y, center_z)
oy = center_y - W / 2.0
oz = center_z - H / 2.0

# Initialisation de la figure
fig, ax = plt.subplots(figsize=(10, 10))

# ==========================================
# 1. Tracé de la section globale en torsion
# ==========================================
# Cercle extérieur représentant le tube
circle_out = plt.Circle((0, 0), R, color='lightgray', alpha=0.3, ec='black', lw=2)
ax.add_patch(circle_out)

# Cercle intérieur représentant le béton coulé (R2)
circle_in = plt.Circle((0, 0), R2, color='darkgray', alpha=0.4, ec='black', lw=1.5, linestyle='--')
ax.add_patch(circle_in)

# Repère global (Y horizontal, Z vertical)
ax.quiver(0, 0, R+5, 0, angles='xy', scale_units='xy', scale=1, color='black', width=0.004)
ax.quiver(0, 0, 0, R+5, angles='xy', scale_units='xy', scale=1, color='black', width=0.004)
ax.text(R+6, 0, 'y', fontsize=24, weight='bold', va='center')
ax.text(0, R+6, 'z', fontsize=24, weight='bold', ha='center')

# Centre O
ax.scatter(0, 0, color='black', s=40, zorder=5)
ax.text(1, -2, 'O', fontsize=18, weight='bold')

# Moment de torsion global Mt (Grand arc orange)
arc_mt = np.linspace(np.pi/4, np.pi/2, 50)
R_mt = R + 2
ax.plot(R_mt*np.cos(arc_mt), R_mt*np.sin(arc_mt), color='darkorange', linewidth=2.5)
# Pointe de la flèche du moment
v_y = -np.sin(arc_mt[-1])
v_z = np.cos(arc_mt[-1])
ax.quiver(R_mt*np.cos(arc_mt[-1]), R_mt*np.sin(arc_mt[-1]), 3*v_y, 3*v_z, 
          angles='xy', scale_units='xy', scale=1, color='darkorange', width=0.005)
ax.text(-R_mt+35, R_mt-2, '$M_t$', color='darkorange', fontsize=24, weight='bold')

# Lignes de repérage pour positionner le prisme
ax.plot([0, 0], [center_z, center_z], 'k:', alpha=0.6)
ax.plot([0, center_y], [0, center_z], 'k:', alpha=0.6)

# ==========================================
# 2. Tracé du prisme élémentaire translaté
# ==========================================
# Le polygone du prisme (décalé de oy et oz)
polygon = plt.Polygon([[oy, oz + H], [oy, oz], [oy + W, oz]], facecolor="#9E9E9E", edgecolor='black', linewidth=1.5)
ax.add_patch(polygon)

# Variables pour le positionnement des flèches locales
f_len = 2.5  # Longueur des flèches locales
mid_y = oy + W / 2.0
mid_z = oz + H / 2.0

# --- Face verticale (gauche) ---
# Normale à l'axe Y : Contrainte normale = sigma_y
ax.quiver(oy, oz + H/2, -f_len, 0, angles='xy', scale_units='xy', scale=1, color='red', width=0.004)
ax.text(oy - f_len - 2.5, oz + H/2, r'$\sigma_y$', color='red', fontsize=24, va='center', weight='bold')

# Cisaillement = tau_yz
ax.quiver(oy, oz + H/2, 0, -f_len, angles='xy', scale_units='xy', scale=1, color='green', width=0.004)
ax.text(oy - 3, oz + H/2 - f_len - 1, r'$\tau_{yz}$', color='green', fontsize=24, weight='bold')

# --- Face horizontale (bas) ---
# Normale à l'axe Z : Contrainte normale = sigma_z
ax.quiver(oy + W/2, oz, 0, -f_len, angles='xy', scale_units='xy', scale=1, color='red', width=0.004)
ax.text(oy + W/2, oz - f_len - 1.5, r'$\sigma_z$', color='red', fontsize=24, ha='center', weight='bold')

# Cisaillement = tau_zy
ax.quiver(oy + W/2, oz, -f_len, 0, angles='xy', scale_units='xy', scale=1, color='green', width=0.004)
ax.text(oy + W/2 - f_len - 1, oz - 2.0, r'$\tau_{zy}$', color='green', fontsize=24, weight='bold')

# --- Face inclinée (Ce qu'on calcule) ---
nx, ny = 1.5 * f_len * np.cos(theta), 1.5 * f_len * np.sin(theta)
ax.quiver(mid_y, mid_z, nx, ny, angles='xy', scale_units='xy', scale=1, color='mediumblue', width=0.006)
ax.text(mid_y + nx + 0.5, mid_z + ny + 0.5, r'$\sigma_{\theta}$', color='mediumblue', fontsize=24, weight='bold')

# --- Tracé de l'angle Theta ---
ax.plot([mid_y, mid_y + 3], [mid_z, mid_z], 'k--', alpha=0.5)
arc_angles = np.linspace(0, theta, 20)
R_arc = 1.5
ax.plot(mid_y + R_arc * np.cos(arc_angles), mid_z + R_arc * np.sin(arc_angles), 'k-', linewidth=1.5)
ax.text(mid_y + R_arc*1.3*np.cos(theta/2), mid_z + R_arc*1.3*np.sin(theta/2), r'$45^\circ$', fontsize=12, weight='bold')

# Symbole d'angle droit sur la face inclinée
ax.plot([mid_y, mid_y - 0.8*np.sin(theta)], [mid_z, mid_z + 0.8*np.cos(theta)], 'k--', alpha=0.5)

# ==========================================
# 3. Représentation linéaire du cisaillement (Papillons)
# ==========================================
max_tau = 4.0  # Longueur maximale des vecteurs cisaillement
pts = np.linspace(-R, R, 15)

# 3A. Sur l'axe Z (vecteurs horizontaux, selon Y)
# Le moment Mt crée un effort vers les -Y pour Z positif, et +Y pour Z négatif
tau_y = -max_tau * (pts / R)
ax.quiver(np.zeros_like(pts), pts, tau_y, np.zeros_like(pts), 
          angles='xy', scale_units='xy', scale=1, color='purple', width=0.003, alpha=0.7)

# Enveloppe de la distribution sur l'axe Z
ax.plot([-max_tau, max_tau], [R, -R], color='purple', linewidth=1.5, linestyle='-')
ax.plot([0, -max_tau], [R, R], color='purple', linewidth=1.5)
ax.plot([0, max_tau], [-R, -R], color='purple', linewidth=1.5)
ax.text(-max_tau - 3, R, r'$\tau_{max}$', color='purple', fontsize=16, weight='bold')

# 3B. Sur l'axe Y (vecteurs verticaux, selon Z)
# Le moment Mt crée un effort vers les +Z pour Y positif, et -Z pour Y négatif
tau_z = max_tau * (pts / R)
ax.quiver(pts, np.zeros_like(pts), np.zeros_like(pts), tau_z, 
          angles='xy', scale_units='xy', scale=1, color='purple', width=0.003, alpha=0.7)

# Enveloppe de la distribution sur l'axe Y
ax.plot([R, -R], [max_tau, -max_tau], color='purple', linewidth=1.5, linestyle='-')
ax.plot([R, R], [0, max_tau], color='purple', linewidth=1.5)
ax.plot([0, 0],[0, -R], color='purple', linewidth=1.5)
ax.plot([-R, 0], [0, 0], color='purple', linewidth=1.5)
ax.plot([-R, -R], [0, -max_tau], color='purple', linewidth=1.5)

# ==========================================
# Paramètres d'affichage
# ==========================================
lim = R + 8
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig("Torsion_traction.pdf", bbox_inches="tight")
plt.show()