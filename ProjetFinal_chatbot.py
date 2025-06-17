#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Création d'une application streamlit capable d'analyser et prédire le total ESG
import pandas as pd
import numpy as np


# In[2]:


# Charger les données
Data = pd.read_csv(r"C:\Users\PC\OneDrive\Bureau\Application carbone\sp500_esg_data.csv", sep = ",")
# Lire la base de données
print(Data.head())


# In[3]:


# Etudier la corrélation des variables par rapport à total ESG
# Sélection des variables numériques
num_vars = [
    'environmentScore', 'socialScore', 'governanceScore',
    'totalEsg', 'highestControversy', 'percentile',
    'marketCap', 'beta', 'overallRisk'
]


# In[4]:


# Calcul de la matrice de corrélation
correlation_matrix = Data[num_vars].corr()
# Affichage des corrélations avec total ESG
correlation_with_risk = correlation_matrix['totalEsg'].sort_values(ascending=False)
print(correlation_with_risk)


# In[5]:


# Résumé statistique de la base de donnée
Data.describe()


# In[6]:


# Vérification de ma base de donnée pour voir si il il y a pas des valeurs manquantes
print(Data.isnull().sum())


# In[7]:


# Sélection des features et cible
features = ["percentile", "environmentScore", "socialScore", "highestControversy", "governanceScore", "beta", "marketCap"]
target = "totalEsg"

X = Data[features]
y = Data[target]


# In[8]:


# Vérifier si il y a des valeyrs manquantes
print(X.isnull().sum())
print(y.isnull().sum())


# In[9]:


# Normalisation des données et la division des données en entrainnement et test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=50)


# ## Définition du réseau de neurone

# In[10]:


# Charger les bibliothèques nécessaires
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ## Faisons l'architécture du réseau

# In[11]:


# Définition du réseau de neurones
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)  # Sortie unique pour prédiction continue
])


# In[12]:


# Compilation du modèle
model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# In[13]:


# Entraînement 
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.1)


# In[14]:


# Prédictions
y_pred = model.predict(X_test)


# In[15]:


# Calcul des métriques
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[16]:


print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R² Score : {r2:.4f}")


# In[17]:


# Graphique des résultats
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Prédictions vs Réel")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label="Idéal")
plt.xlabel("Valeurs Réelles")
plt.ylabel("Prédictions")
plt.legend()
plt.show()


# # Interfaçage de l'application streamlit

# In[18]:


# Interface utilisateur avec Streamlit
# Charger les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[19]:


# Interface utilisateur avec Streamlit
st.title(" Réseau de Neurones pour Prédire `totalEsg`")
st.markdown("""
Cette application présente les performances d'un modèle de prédiction ESG.
Elle compare les **valeurs réelles** et les **valeurs prédites** à l'aide de courbes et de métriques statistiques.
""")


# In[20]:


# Hyperparamètres ajustables
hidden_units = st.slider("Nombre de neurones cachés", 16, 256, 64)
dropout_rate = st.slider("Taux de Dropout", 0.0, 0.5, 0.2)
epochs = st.slider("Nombre d'épochs", 50, 500, 100)


# In[21]:


# Définition du réseau de neurones
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(hidden_units, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(hidden_units//2, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(1)  # Sortie unique pour prédiction continue
])


# In[22]:


# Compilation du modèle
model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# In[23]:


# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)


# In[24]:


# Prédictions
y_pred = model.predict(X_test)


# In[25]:


# Calcul des métriques
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**MAE:** {mae:.4f}")
st.write(f"**MSE:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")


# In[26]:


# Graphique interactif
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Prédictions vs Réel")
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label="Idéal")
ax.set_xlabel("Valeurs Réelles")
ax.set_ylabel("Prédictions")
ax.legend()
st.pyplot(fig)


# In[27]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

# Prédictions du modèle sur les données de test
y_pred = model.predict(X_test)

# Calcul des métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Affichage des métriques
st.subheader("📈 Évaluation du Modèle")
col1, col2, col3 = st.columns(3)
col1.metric("📉 MAE", f"{mae:.4f}")
col2.metric("📈 RMSE", f"{rmse:.4f}")
col3.metric("🧮 R²", f"{r2:.4f}")

with st.expander("ℹ️ Que signifient ces métriques ?"):
    st.markdown("""
    - **MAE (Mean Absolute Error)** : moyenne des erreurs absolues.
    - **RMSE (Root Mean Squared Error)** : comme MAE mais pénalise davantage les grosses erreurs.
    - **R² (Score de détermination)** : 1 = parfait, 0 = pas de relation.
    """)


# In[28]:


import pandas as pd
import plotly.express as px
import streamlit as st

# S'assurer que les deux sont bien des vecteurs 1D (via .to_numpy())
y_test_flat = y_test.to_numpy()
y_pred_flat = y_pred.flatten()  # Utile si y_pred est un tableau numpy 2D

# Créer le DataFrame
df_pred = pd.DataFrame({
    'Valeurs Réelles': y_test_flat,
    'Valeurs Prédites': y_pred_flat
})

# Sous-titre Streamlit
st.subheader("🔍 Visualisation : Prédictions vs Valeurs Réelles")

# Graphique interactif
fig = px.scatter(
    df_pred,
    x='Valeurs Réelles',
    y='Valeurs Prédites',
    title='Prédictions vs Réel (graphique interactif)',
    opacity=0.6
)

# Ligne diagonale idéale
min_val = min(df_pred['Valeurs Réelles'].min(), df_pred['Valeurs Prédites'].min())
max_val = max(df_pred['Valeurs Réelles'].max(), df_pred['Valeurs Prédites'].max())

fig.add_shape(
    type='line',
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(color='red', dash='dash')
)

# Affichage dans Streamlit
st.plotly_chart(fig)


# In[29]:


# Convertir les colonnes en 1D si nécessaire
y_test_flat = y_test.to_numpy().flatten()
y_pred_flat = y_pred.flatten()  # utile si y_pred est de forme (n, 1)

# Création du DataFrame
results_df = pd.DataFrame({
    "Valeur réelle": y_test_flat,
    "Prédiction": y_pred_flat
})

# Export CSV pour Streamlit
csv = results_df.to_csv(index=False).encode("utf-8")

# Bouton de téléchargement
st.download_button(
    label="📥 Télécharger les prédictions (CSV)",
    data=csv,
    file_name="predictions_esg.csv",
    mime="text/csv"
)



# In[30]:


# Sélection dynamique d'entreprises
st.sidebar.title("Sélection d’entreprise")
entreprises = Data["Full Name"].unique()
selection = st.sidebar.multiselect("Choisissez une ou plusieurs entreprises", entreprises)


# In[31]:


# Affichage des scores ESG pour les entreprises sélectionnées
if selection:
    for ent in selection:
        st.subheader(f"Scores ESG pour {ent}")
        data_ent = Data[Data["Full Name"] == ent].iloc[0]
        st.write(f"🌿 Environnement : {data_ent['environmentScore']}")
        st.write(f"🤝 Social : {data_ent['socialScore']}")
        st.write(f"🏛️ Gouvernance : {data_ent['governanceScore']}")
        st.write(f"📊 Score ESG global : {data_ent['totalEsg']}")


# In[32]:


# Comparaison entre entreprises
if len(selection) > 1:
    st.subheader("📊 Comparaison ESG entre entreprises")
    comp_Data = Data[Data["Full Name"].isin(selection)].set_index("Full Name")[["environmentScore", "socialScore", "governanceScore"]]
    # Affichage sous forme de tableau
    st.dataframe(comp_Data)

    # affichage graphique
    st.bar_chart(comp_Data)

else:
    st.info("Veuillez sélectionner au moins deux entreprises pour comparer.")


# In[33]:


import os
import fitz  # pip install pymupdf

def charger_documents_pdf(dossier):
    base_connaissance = {}
    for fichier in os.listdir(dossier):
        if fichier.endswith(".pdf"):
            chemin = os.path.join(dossier, fichier)
            doc = fitz.open(chemin)
            texte = ""
            for page in doc:
                texte += page.get_text()
            base_connaissance[fichier] = texte
    return base_connaissance

# Charger les rapports
chemin_dossier = r"C:\Users\PC\OneDrive\Bureau\Rapport ESG"
documents = charger_documents_pdf(chemin_dossier)


# In[34]:


# Créer un agent conversationnel
from difflib import get_close_matches

def repondre(question, base_connaissance):
    matches = get_close_matches(question, base_connaissance.keys(), n=1, cutoff=0.3)
    if matches:
        return base_connaissance[matches[0]][:1000] + "..."  # extrait limité
    else:
        return "Je n’ai pas trouvé de réponse pertinente dans les documents ESG."


# In[35]:


import streamlit as st

st.title("🤖 Agent ESG local (PDF)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez une question sur les rapports ESG :"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reponse = repondre(prompt, documents)
    st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)


# In[36]:


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🧑‍💻 Application développée par <b>Kenson FAVEUR</b> – 2025<br>
        <i>Projet d'analyse ESG avec intelligence artificielle</i>
    </div>
    """,
    unsafe_allow_html=True
)


# In[ ]:




