

# Réseau de neurones avec Streamlit
# Application ESG – Analyse et Prédiction des Scores ESG avec Streamlit

Ce projet a été réalisé dans le cadre de la formation **Analyste Data Science** à l’Université Paris Cité. Il vise à démocratiser l’accès aux scores ESG (Environnement, Social, Gouvernance) à travers une application interactive développée avec **Streamlit**, enrichie par un **réseau de neurones** pour la prédiction.

##  Objectifs

- Visualiser les performances ESG d'entreprises du S&P 500.
- Comparer les scores environnementaux, sociaux et de gouvernance.
- Prédire les scores ESG globaux à partir de variables financières et sectorielles via un modèle de deep learning.
- Proposer une interface simple et pédagogique.

## 🖥️ Aperçu de l’application

<img width="927" alt="image" src="https://github.com/user-attachments/assets/a3288255-4401-42b5-8936-4ad022975f9b" />

## Données utilisées
Les données proviennent de Kaggle – S&P 500 ESG Risk Ratings Dataset.
Elles contiennent les scores ESG individuels, les controverses, les capitalisations boursières et d'autres indicateurs.

## Modèle de prédiction
Le modèle utilisé est un réseau de neurones (Keras/TensorFlow), entraîné pour prédire le totalEsg à partir de variables comme :

environmentScore

socialScore

governanceScore

marketCap

beta

overallRisk

Métriques de performance :

MAE : ~0.65

MSE : ~0.78

R² : 0.9825
