# 🍳 Recipe Recommendation System with RAG

Un système intelligent de recommandation de recettes basé sur vos ingrédients, utilisant la technique RAG (Retrieval-Augmented Generation) et les embeddings sémantiques.

## 📋 Table des matières

- [Description du projet](#-description-du-projet)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Utilisation](#-utilisation)
- [Comment ça marche](#-comment-ça-marche)
- [Phases du projet](#-phases-du-projet)
- [Exemples](#-exemples)
- [Troubleshooting](#-troubleshooting)
- [Prochaines étapes](#-prochaines-étapes)

## 🎯 Description du projet

Ce projet vous permet de :
- **Entrer des ingrédients** que vous avez chez vous
- **Recevoir des recommandations** de recettes pertinentes
- **Voir les détails complets** : ingrédients, instructions, temps de préparation
- Utiliser la **recherche sémantique** (comprend "tomate" même si vous cherchez "tomato")

### Objectif final
Créer un LLM capable de générer des recettes personnalisées à partir d'une liste d'ingrédients.

## 🏗️ Architecture

```
Dataset Food.com (200k+ recettes)
        ↓
[Phase 1] Exploration et préparation
        ↓
Recettes formatées + Embeddings vectoriels
        ↓
[Phase 2] Système RAG (Recherche sémantique)
        ↓
Recommandations de recettes
        ↓
[Phase 3] Intégration LLM (à venir)
        ↓
Génération de recettes en langage naturel
```

## 🔧 Prérequis

- Python 3.8+
- 4GB de RAM minimum
- 2GB d'espace disque (pour le dataset et les embeddings)

## 📥 Installation

### 1. Cloner le projet

```bash
git clone <votre-repo>
cd recipe-llm-project
```

### 2. Créer un environnement virtuel

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn sentence-transformers scikit-learn
```

**Liste complète des packages :**
- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `matplotlib` / `seaborn` : Visualisations
- `sentence-transformers` : Embeddings sémantiques (modèle all-MiniLM-L6-v2)
- `scikit-learn` : Calcul de similarité cosinus

### 4. Télécharger le dataset

**Option A : Kaggle CLI (recommandé)**
```bash
pip install kaggle
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
unzip food-com-recipes-and-user-interactions.zip -d food-com-recipes-and-user-interactions/
rm -rf food-com-recipes-and-user-interactions.zip
```

**Option B : Manuellement**
1. Aller sur [Kaggle Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
2. Télécharger le dataset
3. Extraire dans le dossier `food-com-recipes-and-user-interactions/`

## 📁 Structure du projet

```
.
├── food-com-recipes-and-user-interactions/
│   ├── RAW_recipes.csv              # Dataset brut (231,637 recettes)
│   ├── RAW_interactions.csv         # Avis et notes des utilisateurs
│   ├── PP_recipes.csv               # Dataset pré-traité
│   └── ...
│
├── recipes/
│   ├── recipes_sample.csv           # Échantillon de 1000 recettes (pour tests rapides)
│   ├── recipes_prepared_sample.csv  # Échantillon formaté pour le LLM
│   ├── recipes_prepared.csv         # Toutes les recettes formatées
│   └── recipes_prepared_sample_embeddings.pkl  # Embeddings pré-calculés (cache)
│
├── phase1_exploration.py            # Script d'exploration et préparation des données
├── phase2_rag.py                    # Système RAG de recommandation
├── README.md                        # Ce fichier
└── requirements.txt                 # Dépendances Python (optionnel)
```

## 🚀 Utilisation

### Phase 1 : Exploration et préparation des données

**But :** Analyser le dataset et créer les fichiers formatés pour le système RAG.

```bash
python phase1_exploration.py
```

**Ce qui se passe :**
1. Charge `RAW_recipes.csv` (231k recettes)
2. Analyse les ingrédients, tags, temps de cuisson
3. Crée `recipes_sample.csv` (1000 recettes pour tests)
4. Crée `recipes_prepared.csv` (toutes les recettes formatées avec `llm_text`)

**Durée :** ~2-5 minutes

**Fichiers générés :**
- `recipes/recipes_sample.csv`
- `recipes/recipes_prepared.csv`

---

### Phase 2 : Système RAG (Recommandation)

**But :** Rechercher des recettes en fonction de vos ingrédients.

```bash
python phase2_rag.py
```

**Interface interactive :**

```
🍳 SYSTÈME DE RECOMMANDATION DE RECETTES
══════════════════════════════════════════════════════════════════════

⚙️  Configuration:
Chemin du fichier recettes (défaut: recipes_sample.csv): recipes/recipes_prepared_sample.csv

📂 Chargement des recettes depuis recipes/recipes_prepared_sample.csv...
✓ 1000 recettes chargées

🔄 Création des embeddings...
[████████████████████████] 1000/1000

══════════════════════════════════════════════════════════════════════

🥕 Entrez vos ingrédients séparés par des virgules (ou 'quit' pour quitter):
> chicken, garlic, lemon

🔍 Recherche de recettes avec: ['chicken', 'garlic', 'lemon']

🍽️  RECETTES RECOMMANDÉES
══════════════════════════════════════════════════════════════════════

#1 - Lemon Garlic Chicken
──────────────────────────────────────────────────────────────────────
📊 Score de similarité: 0.847
✓ Matching exact: 75.0%
  • Ingrédients en commun: 3
  • Ingrédients supplémentaires nécessaires: 2
⏱️  Temps: 30 min

🥕 Ingrédients (5):
   • chicken breast
   • garlic cloves
   • lemon juice
   • olive oil
   • salt

💡 Voir le détail d'une recette ? (numéro ou Enter pour continuer): 1
```

**Première fois :** 2-5 minutes (création des embeddings)  
**Fois suivantes :** < 1 seconde (embeddings en cache)

---

## 🧠 Comment ça marche

### 1. Embeddings : Traduire le texte en vecteurs

Les **embeddings** transforment le texte en vecteurs numériques qui capturent le sens :

```python
"chicken, garlic, lemon" → [0.23, -0.45, 0.12, ..., 0.89]  (384 dimensions)
"poulet, ail, citron"    → [0.25, -0.43, 0.11, ..., 0.87]  (très similaire!)
```

**Modèle utilisé :** `all-MiniLM-L6-v2` de Sentence Transformers
- Léger : 80MB
- Rapide : 1000 phrases/seconde
- Multilingue : Comprend anglais, français, espagnol, etc.

### 2. RAG : Retrieval-Augmented Generation

```
Vos ingrédients
    ↓
Encoder en vecteur
    ↓
Comparer avec 1000 recettes encodées
    ↓
Similarité cosinus → Scores
    ↓
Top 5 recettes
    ↓
Affichage détaillé
```

### 3. Matching intelligent

Le système calcule deux scores :
- **Similarité sémantique** (60%) : Compréhension du sens
- **Matching exact** (40%) : Ingrédients en commun

**Score combiné = 0.6 × similarité + 0.4 × matching**

---

## 📚 Phases du projet

### ✅ Phase 1 : Exploration (Terminée)
- Analyse du dataset Food.com
- Préparation des données
- Création des fichiers formatés

### ✅ Phase 2 : RAG (Terminée)
- Système de recherche sémantique
- Matching d'ingrédients
- Interface CLI

### 🚧 Phase 3 : Intégration LLM (En cours)
- Génération de réponses en langage naturel
- Fine-tuning (optionnel)
- Amélioration de l'expérience utilisateur

### 🔮 Phase 4 : Interface Web (Futur)
- Application Streamlit/Gradio
- API REST
- Déploiement

---

## 💡 Exemples

### Exemple 1 : Recherche simple

```bash
🥕 Entrez vos ingrédients:
> pasta, tomato, garlic

# Résultats: Spaghetti Aglio e Olio, Pasta Pomodoro, etc.
```

### Exemple 2 : Recherche avec filtres

```python
from phase2_rag import RecipeRAGSystem

rag = RecipeRAGSystem()
rag.load_recipes('recipes/recipes_prepared_sample.csv')
rag.load_embeddings('recipes/recipes_prepared_sample_embeddings.pkl')

results = rag.search_with_filters(
    user_ingredients=['chicken', 'rice', 'curry'],
    top_k=5,
    min_match_score=0.5,      # Au moins 50% de matching
    max_cooking_time=30       # Maximum 30 minutes
)

rag.display_results(results)
```

### Exemple 3 : Détail d'une recette

Après avoir tapé `1` pour voir le détail :

```
══════════════════════════════════════════════════════════════════════
🍽️  LEMON GARLIC CHICKEN
══════════════════════════════════════════════════════════════════════

📝 Description:
   A simple and flavorful chicken dish with bright lemon and aromatic
   garlic. Perfect for a quick weeknight dinner.

⏱️  30 min | 📋 5 étapes

📊 Score de similarité: 84.7%
✓ Matching d'ingrédients: 75.0%

🥕 INGRÉDIENTS (5):
──────────────────────────────────────────────────────────────────────
  ✓ 1. chicken breast
  ✓ 2. garlic cloves
  ✓ 3. lemon juice
    4. olive oil
    5. salt and pepper

👨‍🍳 INSTRUCTIONS (5 étapes):
──────────────────────────────────────────────────────────────────────
  1. Season chicken with salt and pepper

  2. Heat olive oil in a large skillet over medium-high heat

  3. Add chicken and cook until golden brown, 5-7 minutes per side

  4. Add minced garlic and cook for 1 minute until fragrant

  5. Add lemon juice and simmer for 2-3 minutes

══════════════════════════════════════════════════════════════════════
```

---

## 🐛 Troubleshooting

### Erreur : "File not found"

**Problème :** Le fichier `recipes_prepared_sample.csv` n'existe pas.

**Solution :**
```bash
# Exécuter d'abord la Phase 1
python phase1_exploration.py
```

### Erreur : "Out of memory"

**Problème :** Pas assez de RAM pour le dataset complet.

**Solution :**
```python
# Utiliser l'échantillon de 1000 recettes
rag.load_recipes('recipes/recipes_prepared_sample.csv')
```

### Résultats non pertinents

**Problème :** Les recettes recommandées ne correspondent pas.

**Solutions :**
1. Utiliser des noms d'ingrédients en **anglais**
2. Être plus spécifique : `chicken breast` plutôt que `chicken`
3. Ajuster le `min_match_score` :
   ```python
   results = rag.search_with_filters(
       user_ingredients=['...'],
       min_match_score=0.3  # Plus permissif (défaut: 0.2)
   )
   ```

### Embeddings très lents

**Problème :** La création des embeddings prend plus de 10 minutes.

**Solutions :**
1. Utiliser `recipes_prepared_sample.csv` (1000 recettes) au lieu du dataset complet
2. Les embeddings sont mis en cache, la 2ème fois sera instantanée
3. Utiliser un GPU si disponible (détection automatique)

### Erreur : "No module named 'sentence_transformers'"

**Solution :**
```bash
pip install sentence-transformers
```

---

## 🔧 Configuration avancée

### Changer le modèle d'embeddings

```python
# Modèle plus précis mais plus lourd (420MB)
rag = RecipeRAGSystem(model_name='all-mpnet-base-v2')

# Modèle par défaut (80MB, rapide)
rag = RecipeRAGSystem(model_name='all-MiniLM-L6-v2')
```

### Ajuster les paramètres de recherche

```python
results = rag.search_with_filters(
    user_ingredients=['...'],
    top_k=10,                    # Nombre de résultats (défaut: 5)
    min_match_score=0.5,         # Score minimum 0-1 (défaut: 0.3)
    max_cooking_time=45          # Temps max en minutes
)
```

---

## 🚀 Prochaines étapes

### Phase 3 : Intégration d'un LLM

Plusieurs options possibles :

**Option A : LLM local (gratuit)**
- LLaMA 2 / Mistral avec Ollama
- GPT4All
- Avantages : Gratuit, privé, offline
- Inconvénient : Nécessite plus de ressources

**Option B : API externe (payant)**
- OpenAI GPT-4
- Anthropic Claude
- Google Gemini
- Avantages : Très performant, rapide
- Inconvénient : Coût par requête

**Option C : Fine-tuning (avancé)**
- Fine-tuner un petit modèle (GPT-2, DistilGPT)
- Sur le dataset Food.com
- Avantages : Optimisé pour les recettes
- Inconvénient : Temps et ressources nécessaires

### Phase 4 : Interface utilisateur

- **Streamlit** : Interface web rapide et simple
- **Gradio** : Démo interactive
- **FastAPI** : API REST pour intégration

---

## 📊 Performances

| Dataset | Recettes | Temps embeddings | Temps recherche |
|---------|----------|------------------|-----------------|
| Sample (1k) | 1,000 | ~30 secondes | < 0.1s |
| Complet (200k) | 231,637 | ~15-30 minutes | < 1s |

**Note :** Les embeddings sont calculés une seule fois et mis en cache.

---

## 📝 Crédits

- **Dataset :** [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) par Shuyang Li
- **Modèle d'embeddings :** [Sentence Transformers](https://www.sbert.net/) par UKPLab
- **Inspirations :** RAG (Retrieval-Augmented Generation) technique

---

## 📄 Licence

Ce projet est à usage éducatif. Le dataset Food.com a sa propre licence (voir Kaggle).

---

## 🤝 Contribution

Des questions ? Des améliorations ? N'hésitez pas à ouvrir une issue ou proposer une pull request !

---

**Bon appétit et bon coding ! 🍽️👨‍💻**