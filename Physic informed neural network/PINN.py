import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
from dolfinx import mesh
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. Génération des points (Domaine FEniCSx)
# ==========================================
domain = mesh.create_box(MPI.COMM_WORLD, 
                         [[0.0, 0.0, 0.0], [5.0, 1.0, 1.0]], 
                         [20, 5, 5])

np.random.seed(42)
X_quad_np = np.random.uniform(0, 5, (5000, 3)) 
W_quad_np = np.ones(5000) * (5.0 / 5000) 

X_quad = torch.tensor(X_quad_np, dtype=torch.float32, requires_grad=True)
W_quad = torch.tensor(W_quad_np, dtype=torch.float32)

X_left = torch.tensor(X_quad_np[X_quad_np[:, 0] < 0.1], dtype=torch.float32)

# ==========================================
# 2. Le Réseau de Neurones Paramétrique
# ==========================================
class Parametric_VPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. La Fonction d'Énergie (Loss Paramétrique)
# ==========================================
def compute_parametric_energy(model, X_quad, W_quad, X_left, force_y):
    # NORMALISATION ICI : On donne au réseau une valeur entre 0 et -1
    force_normalized = force_y / 100.0 
    
    F_quad = torch.full((X_quad.shape[0], 1), force_normalized, dtype=torch.float32)
    F_left = torch.full((X_left.shape[0], 1), force_normalized, dtype=torch.float32)
    
    Inputs_quad = torch.cat([X_quad, F_quad], dim=1)
    Inputs_left = torch.cat([X_left, F_left], dim=1)
    
    # ... la suite de la fonction reste EXACTEMENT identique ...
    # (le calcul de external_work utilise bien le VRAI force_y)
    
    E, nu = 1e5, 0.3
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    
    u_left = model(Inputs_left)
    penalty_energy = 1e6 * torch.mean(u_left**2) 
    
    u = model(Inputs_quad)
    
    du_x = torch.autograd.grad(u[:, 0].sum(), X_quad, create_graph=True)[0]
    du_y = torch.autograd.grad(u[:, 1].sum(), X_quad, create_graph=True)[0]
    du_z = torch.autograd.grad(u[:, 2].sum(), X_quad, create_graph=True)[0]
    
    eps_xx, eps_yy, eps_zz = du_x[:, 0], du_y[:, 1], du_z[:, 2]
    eps_xy = 0.5 * (du_x[:, 1] + du_y[:, 0])
    eps_xz = 0.5 * (du_x[:, 2] + du_z[:, 0])
    eps_yz = 0.5 * (du_y[:, 2] + du_z[:, 1])
    
    trace_eps = eps_xx + eps_yy + eps_zz
    energy_density = 0.5 * lambda_ * trace_eps**2 + mu * (
        eps_xx**2 + eps_yy**2 + eps_zz**2 + 2*(eps_xy**2 + eps_xz**2 + eps_yz**2)
    )
    internal_energy = torch.sum(energy_density * W_quad)
    
    external_work = torch.sum((force_y * u[:, 1]) * W_quad)
    
    total_energy = internal_energy - external_work + penalty_energy
    return total_energy

# ==========================================
# 4. Initialisation, Chargement des Poids
# ==========================================
model = Parametric_VPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

WEIGHTS_FILE = "pinn_parametric_weights.pth"
FORCE_TRAINING = False # Mets sur True si tu veux entraîner même si les poids existent

if os.path.exists(WEIGHTS_FILE):
    print(f"✅ Fichier de poids trouvé : '{WEIGHTS_FILE}'")
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    print("✅ Poids chargés avec succès !")
    
    if not FORCE_TRAINING:
        print("⏭️ Entraînement ignoré (passage direct à l'inférence).")
        DO_TRAIN = False
    else:
        print("▶️ FORCE_TRAINING=True : Reprise de l'entraînement avec les poids chargés.")
        DO_TRAIN = True
else:
    print("⚠️ Aucun fichier de poids trouvé, le modèle part de zéro.")
    DO_TRAIN = True

# ==========================================
# 5. Entraînement avec Critère de Convergence
# ==========================================
if DO_TRAIN:
    print("\nDébut de l'entraînement paramétrique...")
    max_epochs = 10000 # Augmenté car on a un arrêt automatique
    
    # --- Paramètres de convergence (Early Stopping) ---
    patience = 200      # On tolère 200 époques sans amélioration avant de couper
    tolerance = 1e-4    # L'énergie doit baisser d'au moins 0.0001 pour compter comme amélioration
    best_val_energy = float('inf')
    epochs_no_improve = 0
    validation_force = -50.0 # Force fixe utilisée uniquement pour juger la convergence
    best_weights = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Phase 1: Entraînement sur une force aléatoire
        current_force = np.random.uniform(-100.0, 0.0)
        train_energy = compute_parametric_energy(model, X_quad, W_quad, X_left, current_force)
        
        train_energy.backward()
        optimizer.step()
        
        # Phase 2: Vérification de la convergence (toutes les 10 époques pour la performance)
        if epoch % 10 == 0:
            optimizer.zero_grad() # Nettoyage avant évaluation
            
            # Évaluation sur la force fixe
            val_energy = compute_parametric_energy(model, X_quad, W_quad, X_left, validation_force).item()
            
            # L'énergie totale potentielle est minimisée, donc elle descend
            if val_energy < best_val_energy - tolerance:
                best_val_energy = val_energy
                epochs_no_improve = 0
                # On sauvegarde en RAM le meilleur état du réseau
                best_weights = model.state_dict().copy()
            else:
                epochs_no_improve += 10
            
            # Affichage console (optionnel, pour suivre)
            if epoch % 500 == 0 or epochs_no_improve >= patience:
                print(f"Epoch {epoch:04d} | Énergie Train (F={current_force:5.1f}): {train_energy.item():8.2f} | Énergie Val (F={validation_force}): {val_energy:8.4f}")
            
            # Déclenchement de l'arrêt prématuré
            if epochs_no_improve >= patience:
                print(f"\n🛑 Convergence atteinte ! L'énergie sur la validation ne s'améliore plus depuis {patience} époques.")
                print(f"Arrêt prématuré à l'époque {epoch}.")
                # On restaure les poids de la meilleure époque (qui peuvent dater d'il y a {patience} époques)
                model.load_state_dict(best_weights)
                break

    print("\nEntraînement terminé !")
    
    print(f"💾 Sauvegarde des meilleurs poids dans '{WEIGHTS_FILE}'...")
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print("✅ Poids sauvegardés avec succès !")

# ==========================================
# 6. Inférence et Animation 3D
# ==========================================
print("\n--- Phase d'Inférence et Création de l'Animation ---")

with torch.no_grad():
    x = np.linspace(0, 5, 25)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    
    vis_points_np = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    vis_points = torch.tensor(vis_points_np, dtype=torch.float32)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(vis_points_np[:, 0], vis_points_np[:, 1], vis_points_np[:, 2], 
           color='gray', alpha=0.15, label='Repos', s=10)

scatter_def = ax.scatter(vis_points_np[:, 0], vis_points_np[:, 1], vis_points_np[:, 2], 
                         color='blue', s=20, label='Déformée') 

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, 5])
ax.set_ylim([-3, 1])
ax.set_zlim([-0.5, 1.5])
ax.legend()

def update(frame):
    current_force = -100.0 * (frame / 50.0)
    
    with torch.no_grad():
        # NORMALISATION ICI AUSSI : On divise par 100 pour l'inférence
        force_normalized = current_force / 100.0
        
        input_tensor = torch.cat([vis_points, torch.full((vis_points.shape[0], 1), force_normalized)], dim=1)
        disp = model(input_tensor).numpy()
        
    # --- Facteur d'échelle (Optionnel) ---
    # Si E est très grand, la vraie déflexion peut être de quelques millimètres.
    # On multiplie par 2 pour bien l'exagérer visuellement sur le graphique
    scale_factor = 20 
    
    # Position finale = Position initiale + (Déplacement * Échelle)
    def_points = vis_points_np + (disp * scale_factor)
    
    # Mise à jour graphique
    scatter_def._offsets3d = (def_points[:, 0], def_points[:, 1], def_points[:, 2])
    ax.set_title(f"Inférence IA | Force appliquée : {current_force:.1f} N")
    
    return scatter_def,

print("Calcul des images en cours...")
ani = animation.FuncAnimation(fig, update, frames=51, interval=100, blit=False)

gif_filename = "flexion_ia_parametrique_reelle.gif"
ani.save(gif_filename, writer='pillow', fps=15)
print(f"✅ Animation sauvegardée sous '{gif_filename}' !")