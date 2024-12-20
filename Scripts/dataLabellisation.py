import os
import pandas as pd

# Fichiers de configuration
bugs_file_path = r'C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\bug_file_association_2.csv'
metrics_folder_path = r'C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\Merged_Metrics'

# Mapping des priorités à des valeurs numériques
priority_mapping = {
    'Blocker': 5,
    'Critical': 4,
    'Major': 3,
    'Minor': 2,
    'Trivial': 1
}

# Charger les bugs et convertir les priorités en valeurs numériques
bugs_file = pd.read_csv(bugs_file_path)
bugs_file['Bug_Priority'] = bugs_file['Bug_Priority'].apply(lambda x: [priority_mapping[p.strip()] for p in eval(x)] if isinstance(x, str) else [0])

def add_priority_column(metrics_file_path):
    # Charger le fichier de métriques
    metrics_file = pd.read_csv(metrics_file_path)
    metrics_file['Priority'] = 0  # Colonne par défaut pour la priorité

    # Parcourir chaque ligne du fichier des bugs
    for _, bug_row in bugs_file.iterrows():
        bug_versions = str(bug_row['Affected Versions']).split(",")
        bug_path = str(bug_row['File_Path']).split("/").pop()
        bug_priorities = bug_row['Bug_Priority']

        # Associer la priorité au fichier et à la version correspondante
        for version, priority in zip(bug_versions, bug_priorities):
            metrics_file.loc[
                (metrics_file['Version'].str.endswith(version)) & (metrics_file['Name'] == bug_path),
                'Priority'
            ] = priority

    # Sauvegarder le fichier de métriques avec la colonne 'Priority' mise à jour
    metrics_file.to_csv(metrics_file_path, index=False)

# Appliquer la mise à jour des priorités pour tous les fichiers de métriques
if __name__ == '__main__':
    for filename in os.listdir(metrics_folder_path):
        metrics_file_path = os.path.join(metrics_folder_path, filename)
        add_priority_column(metrics_file_path)
