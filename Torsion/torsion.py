import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Paramètres de la poutre
# ==========================================
L = 10.0         # Longueur de la poutre
R = 1.0          # Rayon extérieur
R_core = 0.8     # Rayon du matériau au cœur
alpha_deg = 30   # Angle de torsion max en bout de poutre (en degrés)
alpha = np.radians(alpha_deg)

# Création de la figure 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Grilles pour les cylindres
x = np.linspace(0, L, 50)
theta = np.linspace(0, 2 * np.pi, 50)
Xc, Thetac = np.meshgrid(x, theta)

# ==========================================
# 1. Encastrement (Mur à x = 0 et hachures)
# ==========================================
wall_lim = R * 1.8
y_wall = np.linspace(-wall_lim, wall_lim, 10)
z_wall = np.linspace(-wall_lim, wall_lim, 10)
Yw, Zw = np.meshgrid(y_wall, z_wall)
Xw = np.zeros_like(Yw)
ax.plot_surface(Xw, Yw, Zw, alpha=0.2, color='gray')

# Ajout des traits parallèles (hachures d'encastrement) vers les x négatifs
hatch_length = 0.5
for yh in [-wall_lim, wall_lim]:
    for zh in np.linspace(-wall_lim, wall_lim, 9):
        # Lignes obliques partant du mur (x=0) vers l'arrière
        ax.plot([0, -hatch_length], [yh, yh - hatch_length/2], [zh, zh - hatch_length/2], 
                color='dimgray', linewidth=1.2, zorder=0)

ax.text(0, 0, wall_lim + 0.4, "Encastrement", color='black', ha='center', fontsize=12)

# ==========================================
# 2. Cœur de la poutre (Matériau 2)
# ==========================================
Y_core = R_core * np.cos(Thetac)
Z_core = R_core * np.sin(Thetac)
ax.plot_surface(Xc, Y_core, Z_core, alpha=0.4, color='gold', edgecolors='none')

# Face libre du cœur
r_c = np.linspace(0, R_core, 10)
Rc_core, Thetac2 = np.meshgrid(r_c, theta)
ax.plot_surface(np.full_like(Rc_core, L), Rc_core * np.cos(Thetac2), Rc_core * np.sin(Thetac2), alpha=0.4, color='gold')

# ==========================================
# 3. Extérieur de la poutre (Matériau 1)
# ==========================================
Y_out = R * np.cos(Thetac)
Z_out = R * np.sin(Thetac)
ax.plot_surface(Xc, Y_out, Z_out, alpha=0.2, color='cyan', edgecolors='none')

# Face libre extérieure
r_o = np.linspace(R_core, R, 10)
Rc_out, Thetac3 = np.meshgrid(r_o, theta)
ax.plot_surface(np.full_like(Rc_out, L), Rc_out * np.cos(Thetac3), Rc_out * np.sin(Thetac3), alpha=0.5, color='dodgerblue')

# ==========================================
# 4. Annotation de la longueur L
# ==========================================
offset = R * 1.6
ax.plot([0, L], [-offset, -offset], [-offset, -offset], color='black', linewidth=1.5)
ax.plot([0, 0], [-offset-0.2, -offset+0.2], [-offset, -offset], color='black', linewidth=1.5)
ax.plot([L, L], [-offset-0.2, -offset+0.2], [-offset, -offset], color='black', linewidth=1.5)
ax.text(L/2, -offset, -offset - 0.4, 'L', color='black', fontsize=14, weight='bold', ha='center')

# ==========================================
# 5. Déformation de torsion (Fibre et section depuis l'axe Z)
# ==========================================
# Fibre génératrice non déformée sur l'axe Z (pointillés)
ax.plot([0, L], [0, 0], [R, R], color='black', linestyle='--', linewidth=1.5)
# Rayon non déformé en bout sur l'axe Z (pointillés)
ax.plot([L, L], [0, 0], [0, R], color='black', linestyle='--', linewidth=1.5)

# Fibre génératrice déformée (Hélice rouge) - Départ à pi/2
phi_x = (x / L) * alpha
Y_fiber = R * np.cos(np.pi/2 + phi_x)
Z_fiber = R * np.sin(np.pi/2 + phi_x)
ax.plot(x, Y_fiber, Z_fiber, color='red', linewidth=2.5, label='Fibre déformée')

# Rayon déformé en bout
ax.plot([L, L], [0, R*np.cos(np.pi/2 + alpha)], [0, R*np.sin(np.pi/2 + alpha)], color='red', linewidth=2.5)

# Arc de cercle pour l'angle alpha (depuis l'axe Z, donc pi/2)
arc_alpha = np.linspace(np.pi/2, np.pi/2 + alpha, 20)
R_alpha = R * 1.15
ax.plot(np.full_like(arc_alpha, L), R_alpha*np.cos(arc_alpha), R_alpha*np.sin(arc_alpha), color='red', linewidth=2)
ax.text(L, R_alpha*np.cos(np.pi/2 + alpha/2), R_alpha*np.sin(np.pi/2 + alpha/2) + 0.15, r'$\alpha$', color='red', fontsize=16, weight='bold')

# ==========================================
# 6. Repère global au centre de la face libre
# ==========================================
axis_len = 2.5
ax.quiver(L, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
ax.text(L + axis_len + 0.2, 0, 0, 'x', color='r', fontsize=14, weight='bold')

ax.quiver(L, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
ax.text(L, axis_len + 0.2, 0, 'y', color='g', fontsize=14, weight='bold')

ax.quiver(L, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.1, linewidth=2)
ax.text(L, 0, axis_len + 0.2, 'z', color='b', fontsize=14, weight='bold')

# ==========================================
# 7. Moment de torsion Mt
# ==========================================
# Arc placé en bas à droite (de -60° à +30°)
phi_mt = np.linspace(-np.pi/3, np.pi/6, 300) 
R_mt = R * 1.4
X_mt = np.full_like(phi_mt, L)
Y_mt = R_mt * np.cos(phi_mt)
Z_mt = R_mt * np.sin(phi_mt)

ax.plot(X_mt, Y_mt, Z_mt, color='darkorange', linewidth=3)

# Pointe de flèche du moment
v = -np.sin(phi_mt[-1]) 
w = np.cos(phi_mt[-1])  
norm = np.sqrt(v**2 + w**2)
v, w = v/norm, w/norm

ax.quiver(L, Y_mt[-1], Z_mt[-1], 0, v, w, color='darkorange', length=0.1, arrow_length_ratio=2.5, linewidth=3)
ax.text(L, R_mt*np.cos(-np.pi/12) + 0.2, R_mt*np.sin(-np.pi/12) - 0.2, 'Mt', color='darkorange', fontsize=16, weight='bold')

# ==========================================
# 8. Annotation des rayons R1 (interne) et R2 (externe)
# ==========================================
# Angles choisis côté "gauche" pour ne pas croiser les axes Y, Z ou l'angle alpha
theta_r1 = 3 * np.pi / 4  # 135° (en haut à gauche)
theta_r2 = 5 * np.pi / 4  # 225° (en bas à gauche)

# Point d'origine au centre
ax.scatter([L], [0], [0], color='black', s=15, zorder=5)

# Ligne et texte pour R1 (coeur)
y_r1 = R_core * np.cos(theta_r1)
z_r1 = R_core * np.sin(theta_r1)
ax.quiver(L, 0, 0,0, y_r1, z_r1, color='black', arrow_length_ratio=0.1, linewidth=2)
#ax.plot([L, L], [0, y_r1], [0, z_r1], color='black', linestyle='-', linewidth=2)
ax.text(L, y_r1/2 - 0.25, z_r1/2 - 0.2, '$R_1$', color='black', fontsize=14, weight='bold')

# Ligne et texte pour R2 (extérieur)
y_r2 = R * np.cos(theta_r2)
z_r2 = R * np.sin(theta_r2)

ax.quiver(L, 0, 0,0, y_r2, z_r2,color='black', arrow_length_ratio=0.1, linewidth=2)
#ax.plot([L, L], [0, y_r2], [0, z_r2], color='black', linestyle='-', linewidth=2)
ax.text(L, y_r2/2 - 0.1, z_r2/2 - 0.25, '$R_2$', color='black', fontsize=14, weight='bold')

# ==========================================
# Paramètres d'affichage
# ==========================================
ax.set_box_aspect([L, 2*R*1.8, 2*R*1.8], zoom=1.4)
ax.set_axis_off()

ax.view_init(elev=10, azim=-15)

# Affichage
plt.savefig("Torsion.pdf", bbox_inches="tight")
plt.show()