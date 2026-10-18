import subprocess
import os

def compresser_images_conserver_texte(chemin_entree, chemin_sortie, dpi=100):
    try:
        taille_initiale = os.path.getsize(chemin_entree) / (1024 * 1024)
        print(f"Compression agressive des images de {chemin_entree} ({taille_initiale:.2f} Mo)...")

        commande = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            # Options générales
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            # Forcer la compression des images en couleur
            "-dDownsampleColorImages=true",
            "-dColorImageDownsampleType=/Bicubic",
            f"-dColorImageResolution={dpi}",
            # Forcer la compression des images en niveaux de gris
            "-dDownsampleGrayImages=true",
            "-dGrayImageDownsampleType=/Bicubic",
            f"-dGrayImageResolution={dpi}",
            # Forcer la compression des images monochromes
            "-dDownsampleMonoImages=true",
            "-dMonoImageDownsampleType=/Bicubic",
            f"-dMonoImageResolution={dpi}",
            # Fichier de sortie et d'entrée
            f"-sOutputFile={chemin_sortie}",
            chemin_entree
        ]

        subprocess.run(commande, check=True)

        taille_finale = os.path.getsize(chemin_sortie) / (1024 * 1024)
        reduction = (1 - (taille_finale / taille_initiale)) * 100
        
        print(f"Succès ! Fichier sauvegardé : {chemin_sortie} ({taille_finale:.2f} Mo)")
        print(f"Réduction de taille : {reduction:.1f}%")

    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de Ghostscript : {e}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    fichier_original = "these.pdf"
    fichier_compresse = "ma_these_texte_net_images_compressees.pdf"
    
    # Vous pouvez tester 72 DPI pour une compression encore plus forte, ou 150 pour un peu plus de qualité
    compresser_images_conserver_texte(fichier_original, fichier_compresse, dpi=100)