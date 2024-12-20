import os
import pandas as pd

# Dossiers des métriques
old_metrics_dir = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\Clean_Metrics"
new_metrics_dir = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\versions"
output_dir = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\Merged_Metrics"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Lister les fichiers dans chaque dossier
old_metrics_files = {
    f.split("result-metrics-")[-1].split(".csv")[0]: os.path.join(old_metrics_dir, f) 
    for f in os.listdir(old_metrics_dir) if f.endswith(".csv")
}
new_metrics_files = {
    f.split(".csv")[0]: os.path.join(new_metrics_dir, f)
    for f in os.listdir(new_metrics_dir) if f.endswith(".csv")
}

# Afficher les clés des deux dictionnaires
print("Anciennes métriques disponibles :")
print(old_metrics_files.keys())

print("\nNouvelles métriques disponibles :")
print(new_metrics_files.keys())

# Identifier les versions communes
common_versions = set(old_metrics_files.keys()).intersection(new_metrics_files.keys())

# Afficher les versions communes
print("\nVersions communes trouvées :")
print(common_versions)

if not common_versions:
    print("Aucune version commune trouvée entre les anciennes et nouvelles métriques.")
else:
    # Fusionner les métriques pour chaque version
    for version in common_versions:
        print(f"Fusion des métriques pour la version : {version}")

        # Charger les fichiers CSV
        old_metrics_path = old_metrics_files[version]
        new_metrics_path = new_metrics_files[version]

        old_metrics = pd.read_csv(old_metrics_path)
        new_metrics = pd.read_csv(new_metrics_path)

        # Vérifier les colonnes nécessaires
        if "Name" not in old_metrics.columns or "file" not in new_metrics.columns:
            print(f"Colonnes nécessaires manquantes pour la version : {version}")
            continue

        # Extraire uniquement le nom des fichiers dans les nouvelles métriques
        new_metrics["BaseFileName"] = new_metrics["file"].apply(lambda x: os.path.basename(x))

        # Fusionner les métriques
        merged_metrics = pd.merge(
            old_metrics,
            new_metrics,
            how="outer",
            left_on="Name",  # Colonne des anciennes métriques
            right_on="BaseFileName"  # Colonne nettoyée des nouvelles métriques
        )

        # Sauvegarder les métriques fusionnées
        output_file = os.path.join(output_dir, f"{version}.csv")
        merged_metrics.to_csv(output_file, index=False)
        print(f"Métriques fusionnées sauvegardées dans : {output_file}")
