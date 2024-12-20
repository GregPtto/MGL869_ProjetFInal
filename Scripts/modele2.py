import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Créer le dossier pour les résultats
results_folder = "model_results_severity_2.0.0"
os.makedirs(results_folder, exist_ok=True)

# Charger les données
data = pd.read_csv(r'C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\Merged_Metrics\release-2.0.0.csv')

# Pré-traitement des données
data = data.replace({',': '.'}, regex=True)

# Vérification que toutes les colonnes numériques sont de type float
for column in data.select_dtypes(include='object').columns:
    try:
        data[column] = data[column].astype(float)
    except ValueError:
        pass

# Remplacer les NaN et valeurs infinies par 0 ou une valeur par défaut
data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

# Analyser la répartition des classes
print("Répartition des classes :")
print(data['Priority'].value_counts())

# Filtrer les classes extrêmement rares (classe avec seulement 1 échantillon)
#data = data[data['Priority'] != 5]

# Sélectionner les caractéristiques (toutes les colonnes sauf celles inutiles)
X = data.drop(columns=['Priority', 'Kind', 'Name', 'Version','Bogue', 'CommitId', 'BaseFileName_x', 'BaseFileName_y', 'file'])

# Variable cible
y = data['Priority']

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Combiner X et y pour manipuler les données
data_combined = pd.concat([pd.DataFrame(X_scaled), y.reset_index(drop=True)], axis=1)
data_combined.columns = list(X.columns) + ['Priority']

# Rééquilibrer les classes rares
majority_class = data_combined[data_combined['Priority'] == 0]
minority_classes = data_combined[data_combined['Priority'].isin([5, 2, 3, 4, 1])]

# Suréchantillonner les classes minoritaires
oversampled_minority = resample(minority_classes,
                                 replace=True,
                                 n_samples=majority_class.shape[0] // 2, 
                                 random_state=42)

# Combiner les classes rééquilibrées
data_balanced = pd.concat([majority_class, oversampled_minority])

# Réassigner X_scaled et y
X_scaled = data_balanced.drop(columns=['Priority']).values
y = data_balanced['Priority'].values

# Diviser les données en ensembles d'entraîment et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Initialiser les modèles
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
log_reg_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

# Entraîner les modèles
rf_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

# Prédictions
rf_predictions = rf_model.predict(X_test)
log_reg_predictions = log_reg_model.predict(X_test)

# Évaluation des performances
rf_report = classification_report(y_test, rf_predictions, zero_division=0)
log_reg_report = classification_report(y_test, log_reg_predictions, zero_division=0)

# Sauvegarder les rapports
with open(os.path.join(results_folder, "rf_classification_report.txt"), "w") as rf_file:
    rf_file.write(rf_report)

with open(os.path.join(results_folder, "log_reg_classification_report.txt"), "w") as log_reg_file:
    log_reg_file.write(log_reg_report)

# Matrice de confusion pour Random Forest
rf_cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap="Blues")
plt.title("Matrice de Confusion - Random Forest")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.savefig(os.path.join(results_folder, "rf_confusion_matrix.png"))
plt.close()

# Matrice de confusion pour Logistic Regression
log_reg_cm = confusion_matrix(y_test, log_reg_predictions)
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap="Blues")
plt.title("Matrice de Confusion - Logistic Regression")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.savefig(os.path.join(results_folder, "log_reg_confusion_matrix.png"))
plt.close()

# Sauvegarder les importances des variables pour Random Forest
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
rf_importances.to_csv(os.path.join(results_folder, "rf_feature_importances.csv"), index=False)

# Visualisation des importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importances.head(10), palette='viridis')
plt.title("Top 10 Features - Random Forest")
plt.savefig(os.path.join(results_folder, "rf_feature_importances.png"))
plt.close()
