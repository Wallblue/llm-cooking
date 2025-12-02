import pandas as pd
from collections import Counter

# ============================================
# CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================

DATASET_FOLDER = "food-com-recipes-and-user-interactions/"
RECIPES_FOLDER = "recipes/"

class RecipeDatasetExplorer:
    def __init__(self, recipes_path, interactions_path=None):
        """
        Initialise l'explorateur avec les chemins des fichiers CSV
        
        Args:
            recipes_path: chemin vers RAW_recipes.csv
            interactions_path: chemin vers RAW_interactions.csv (optionnel)
        """
        print("📂 Chargement des données...")
        self.recipes_df = pd.read_csv(recipes_path)
        
        if interactions_path:
            self.interactions_df = pd.read_csv(interactions_path)
        else:
            self.interactions_df = None
        
        print(f"✓ {len(self.recipes_df)} recettes chargées")
        if self.interactions_df is not None:
            print(f"✓ {len(self.interactions_df)} interactions chargées")
    
    def explore_basic_info(self):
        """Affiche les informations de base sur le dataset"""
        print("\n" + "="*60)
        print("📊 INFORMATIONS GÉNÉRALES")
        print("="*60)
        
        print(f"\n📌 Colonnes disponibles:")
        for col in self.recipes_df.columns:
            print(f"  - {col}")
        
        print(f"\n📏 Shape: {self.recipes_df.shape}")
        print(f"\n🔍 Aperçu des premières lignes:")
        print(self.recipes_df.head(3))
        
        print(f"\n❓ Valeurs manquantes:")
        print(self.recipes_df.isnull().sum())
        
        return self.recipes_df.info()
    
    def analyze_ingredients(self):
        """Analyse les ingrédients du dataset"""
        print("\n" + "="*60)
        print("🥕 ANALYSE DES INGRÉDIENTS")
        print("="*60)
        
        # Parser les ingrédients (souvent stockés comme string de liste)
        def parse_ingredients(ing_str):
            try:
                # Si c'est déjà une liste
                if isinstance(ing_str, list):
                    return ing_str
                # Si c'est une string représentant une liste
                return eval(ing_str)
            except:
                return []
        
        self.recipes_df['ingredients_parsed'] = self.recipes_df['ingredients'].apply(parse_ingredients)
        
        # Nombre d'ingrédients par recette
        self.recipes_df['n_ingredients'] = self.recipes_df['ingredients_parsed'].apply(len)
        
        print(f"\n📊 Statistiques sur les ingrédients:")
        print(f"  - Moyenne d'ingrédients par recette: {self.recipes_df['n_ingredients'].mean():.2f}")
        print(f"  - Min: {self.recipes_df['n_ingredients'].min()}")
        print(f"  - Max: {self.recipes_df['n_ingredients'].max()}")
        print(f"  - Médiane: {self.recipes_df['n_ingredients'].median()}")
        
        # Top ingrédients
        all_ingredients = []
        for ingredients in self.recipes_df['ingredients_parsed']:
            all_ingredients.extend(ingredients)
        
        ingredient_counts = Counter(all_ingredients)
        print(f"\n🔝 Top 20 ingrédients les plus utilisés:")
        for ing, count in ingredient_counts.most_common(20):
            print(f"  {count:6d}x  {ing}")
        
        return ingredient_counts
    
    def analyze_recipes(self):
        """Analyse les caractéristiques des recettes"""
        print("\n" + "="*60)
        print("⏱️  ANALYSE DES RECETTES")
        print("="*60)
        
        # Temps de préparation
        if 'minutes' in self.recipes_df.columns:
            print(f"\n⏰ Temps de préparation (minutes):")
            print(self.recipes_df['minutes'].describe())
            
            # Filtrer les valeurs aberrantes
            reasonable_time = self.recipes_df[self.recipes_df['minutes'] < 500]
            print(f"\n  Recettes < 500 min: {len(reasonable_time)} ({len(reasonable_time)/len(self.recipes_df)*100:.1f}%)")
        
        # Étapes de préparation
        if 'n_steps' in self.recipes_df.columns:
            print(f"\n📝 Nombre d'étapes:")
            print(self.recipes_df['n_steps'].describe())
        
        # Tags/Catégories
        if 'tags' in self.recipes_df.columns:
            def parse_tags(tag_str):
                try:
                    return eval(tag_str) if isinstance(tag_str, str) else tag_str
                except:
                    return []
            
            self.recipes_df['tags_parsed'] = self.recipes_df['tags'].apply(parse_tags)
            all_tags = []
            for tags in self.recipes_df['tags_parsed']:
                all_tags.extend(tags)
            
            tag_counts = Counter(all_tags)
            print(f"\n🏷️  Top 15 tags:")
            for tag, count in tag_counts.most_common(15):
                print(f"  {count:6d}x  {tag}")
    
    def create_sample_dataset(self, n_samples=1000, output_path=f'{RECIPES_FOLDER}recipes_sample.csv'):
        """Crée un échantillon de données pour tester rapidement"""
        print(f"\n📦 Création d'un échantillon de {n_samples} recettes...")
        
        # Échantillonner
        sample_df = self.recipes_df.sample(n=min(n_samples, len(self.recipes_df)), random_state=42)
        
        # Sauvegarder
        sample_df.to_csv(output_path, index=False)
        print(f"✓ Échantillon sauvegardé dans {output_path}")
        
        return sample_df
    
    def prepare_for_llm(self, output_path=f'{RECIPES_FOLDER}recipes_prepared.csv', n_samples=None):
        """Prépare les données pour l'entraînement d'un LLM"""
        print("\n" + "="*60)
        print("🤖 PRÉPARATION POUR LE LLM")
        print("="*60)
        
        # Créer un format texte structuré
        def format_recipe_for_llm(row):
            try:
                ingredients = eval(row['ingredients']) if isinstance(row['ingredients'], str) else row['ingredients']
                steps = eval(row['steps']) if 'steps' in row and isinstance(row['steps'], str) else []
                
                text = f"### Recette: {row['name']}\n\n"
                text += f"**Ingrédients:**\n"
                for ing in ingredients:
                    text += f"- {ing}\n"
                
                if steps:
                    text += f"\n**Instructions:**\n"
                    for i, step in enumerate(steps, 1):
                        text += f"{i}. {step}\n"
                
                if 'description' in row and pd.notna(row['description']):
                    text += f"\n**Description:** {row['description']}\n"
                
                return text
            except Exception as e:
                return ""
        
        print("📝 Formatage des recettes...")
        self.recipes_df['llm_text'] = self.recipes_df.apply(format_recipe_for_llm, axis=1)
        
        # Sauvegarder
        output_df = self.recipes_df[['id', 'name', 'ingredients', 'steps', 'description', 'llm_text']]
        output_df.to_csv(output_path, index=False)
        
        print(f"✓ Données préparées sauvegardées dans {output_path}")
        print(f"\nExemple de format:")
        print(output_df['llm_text'].iloc[0][:500] + "...")
        
        if n_samples:
            output_df_sample = output_df.sample(n=min(n_samples, len(output_df)), random_state=42)
            sample_path = output_path.replace('.csv', '_sample.csv')
            output_df_sample.to_csv(sample_path, index=False)
            print(f"✓ Échantillon de données préparées sauvegardé dans {sample_path}")

        return output_df


# ============================================
# UTILISATION
# ============================================

if __name__ == "__main__":
    # IMPORTANT: Remplacez par vos vrais chemins de fichiers
    RECIPES_PATH = f"{DATASET_FOLDER}RAW_recipes.csv"
    INTERACTIONS_PATH = f"{DATASET_FOLDER}RAW_interactions.csv"
    
    # Initialiser l'explorateur
    explorer = RecipeDatasetExplorer(
        recipes_path=RECIPES_PATH,
        interactions_path=None  # Mettre INTERACTIONS_PATH si vous l'avez
    )
    
    # Étape 1: Explorer les données
    explorer.explore_basic_info()
    
    # Étape 2: Analyser les ingrédients
    ingredient_counts = explorer.analyze_ingredients()
    
    # Étape 3: Analyser les recettes
    explorer.analyze_recipes()
    
    # Étape 4: Créer un échantillon pour tester
    sample = explorer.create_sample_dataset(n_samples=1000)
    
    # Étape 5: Préparer les données pour le LLM
    prepared_data = explorer.prepare_for_llm(n_samples=1000)
    
    print("\n" + "="*60)
    print("✅ CHARGEMENT ET EXPLORATION DES DONNÉES TERMINÉE !")
    print("="*60)