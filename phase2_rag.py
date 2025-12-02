import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Dict

# ============================================
# PHASE 2 : SYSTÈME RAG - MATCHING D'INGRÉDIENTS
# ============================================

RECIPES_FOLDER = "recipes/"

class RecipeRAGSystem:
    """
    Système RAG pour recommander des recettes basées sur des ingrédients
    RAG = Retrieval-Augmented Generation
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialise le système RAG
        
        Args:
            model_name: Modèle de sentence transformers à utiliser
                'all-MiniLM-L6-v2' = rapide, léger (80MB)
                'all-mpnet-base-v2' = meilleur qualité mais plus lourd
        """
        print("🤖 Initialisation du système RAG...")
        print(f"   Modèle: {model_name}")
        
        # Charger le modèle d'embeddings
        self.model = SentenceTransformer(model_name)
        
        self.recipes_df = None
        self.embeddings = None
        self.ingredient_index = {}
        
        print("✓ Système initialisé")
    
    def load_recipes(self, recipes_path=f'{RECIPES_FOLDER}recipes_prepared_sample.csv'):
        """Charge les recettes préparées"""
        print(f"\n📂 Chargement des recettes depuis {recipes_path}...")
        self.recipes_df = pd.read_csv(recipes_path)
        
        # Parser les ingrédients si nécessaire
        def parse_ingredients(ing):
            try:
                if isinstance(ing, str):
                    return eval(ing)
                return ing
            except:
                return []
        
        if 'ingredients' in self.recipes_df.columns:
            self.recipes_df['ingredients_list'] = self.recipes_df['ingredients'].apply(parse_ingredients)
        
        if 'steps' in self.recipes_df.columns:
            self.recipes_df['steps_list'] = self.recipes_df['steps'].apply(parse_ingredients)
        
        print(f"✓ {len(self.recipes_df)} recettes chargées")
        return self.recipes_df
    
    def create_embeddings(self, save_path='recipe_embeddings.pkl'):
        """
        Crée les embeddings vectoriels de toutes les recettes
        Cette étape peut prendre du temps sur le dataset complet
        """
        print("\n🔄 Création des embeddings...")
        print("   ⚠️  Cela peut prendre plusieurs minutes sur le dataset complet")
        
        # Créer un texte enrichi pour chaque recette
        def create_search_text(row):
            text_parts = []
            
            # Nom de la recette
            if 'name' in row and pd.notna(row['name']):
                text_parts.append(f"Recette: {row['name']}")
            
            # Ingrédients (le plus important !)
            if 'ingredients_list' in row and row['ingredients_list']:
                ingredients_text = ", ".join(row['ingredients_list'])
                text_parts.append(f"Ingrédients: {ingredients_text}")
            
            # Tags/catégories si disponibles
            if 'tags' in row and pd.notna(row['tags']):
                try:
                    tags = eval(row['tags']) if isinstance(row['tags'], str) else row['tags']
                    if tags:
                        text_parts.append(f"Catégories: {', '.join(tags[:5])}")
                except:
                    pass
            
            return " | ".join(text_parts)
        
        self.recipes_df['search_text'] = self.recipes_df.apply(create_search_text, axis=1)
        
        # Créer les embeddings
        texts = self.recipes_df['search_text'].tolist()
        
        print(f"   Encodage de {len(texts)} recettes...")
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Sauvegarder les embeddings
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'recipe_ids': self.recipes_df['id'].tolist() if 'id' in self.recipes_df.columns else list(range(len(self.recipes_df)))
            }, f)
        
        print(f"✓ Embeddings créés et sauvegardés dans {save_path}")
        print(f"   Shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def load_embeddings(self, embeddings_path='recipe_embeddings.pkl'):
        """Charge des embeddings pré-calculés (plus rapide)"""
        print(f"\n📂 Chargement des embeddings depuis {embeddings_path}...")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
        
        print(f"✓ Embeddings chargés: {self.embeddings.shape}")
        return self.embeddings
    
    def search_recipes(self, user_ingredients: List[str], top_k=5) -> List[Dict]:
        """
        Recherche les meilleures recettes pour les ingrédients donnés
        
        Args:
            user_ingredients: Liste d'ingrédients (ex: ['tomato', 'pasta', 'garlic'])
            top_k: Nombre de recettes à retourner
        
        Returns:
            Liste de dictionnaires avec les recettes recommandées
        """
        # Créer la requête
        query = f"Recette avec ingrédients: {', '.join(user_ingredients)}"
        
        # Encoder la requête
        query_embedding = self.model.encode([query])[0]
        
        # Calculer la similarité avec toutes les recettes
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Obtenir les indices des meilleures recettes
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Construire les résultats
        results = []
        for idx in top_indices:
            recipe = self.recipes_df.iloc[idx]
            
            result = {
                'name': recipe.get('name', 'Sans nom'),
                'similarity_score': float(similarities[idx]),
                'ingredients': recipe.get('ingredients_list', []),
                'steps': recipe.get('steps_list', []),
                'description': recipe.get('description', ''),
                'recipe_text': recipe.get('llm_text', ''),
            }
            
            # Ajouter des infos supplémentaires si disponibles
            if 'minutes' in recipe:
                result['cooking_time'] = recipe['minutes']
            if 'n_steps' in recipe:
                result['n_steps'] = recipe['n_steps']
            
            results.append(result)
        
        return results
    
    def calculate_ingredient_match(self, user_ingredients: List[str], recipe_ingredients: List[str]) -> Dict:
        """
        Calcule le matching exact entre ingrédients utilisateur et recette
        
        Returns:
            Dict avec score de matching et détails
        """
        user_set = set([ing.lower().strip() for ing in user_ingredients])
        recipe_set = set([ing.lower().strip() for ing in recipe_ingredients])
        
        # Ingrédients en commun
        matching = user_set.intersection(recipe_set)
        
        # Ingrédients manquants dans la recette
        missing_in_recipe = user_set - recipe_set
        
        # Ingrédients supplémentaires nécessaires
        additional_needed = recipe_set - user_set
        
        # Score de matching
        if len(recipe_set) > 0:
            match_score = len(matching) / len(recipe_set)
        else:
            match_score = 0
        
        return {
            'match_score': match_score,
            'matching_ingredients': list(matching),
            'missing_from_user': list(missing_in_recipe),
            'additional_needed': list(additional_needed),
            'coverage': len(matching) / len(user_set) if len(user_set) > 0 else 0
        }
    
    def search_with_filters(self, 
        user_ingredients: List[str], 
        top_k=10,
        min_match_score=0.3,
        max_cooking_time=None) -> List[Dict]:
        """
        Recherche avancée avec filtres
        """
        # Recherche initiale
        results = self.search_recipes(user_ingredients, top_k=top_k*2)
        
        # Enrichir avec le score de matching exact
        enriched_results = []
        for result in results:
            match_info = self.calculate_ingredient_match(
                user_ingredients,
                result['ingredients']
            )
            result['match_info'] = match_info
            
            # Appliquer les filtres
            if match_info['match_score'] < min_match_score:
                continue
            
            if max_cooking_time and 'cooking_time' in result:
                if result['cooking_time'] > max_cooking_time:
                    continue
            
            enriched_results.append(result)
        
        # Trier par score combiné (similarité sémantique + matching exact)
        for result in enriched_results:
            result['combined_score'] = (
                result['similarity_score'] * 0.6 + 
                result['match_info']['match_score'] * 0.4
            )
        
        enriched_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return enriched_results[:top_k]
    
    def display_results(self, results: List[Dict]):
        """Affiche les résultats de manière lisible"""
        print("\n" + "="*70)
        print("🍽️  RECETTES RECOMMANDÉES")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─'*70}")
            print(f"#{i} - {result['name']}")
            print(f"{'─'*70}")
            print(f"📊 Score de similarité: {result['similarity_score']:.3f}")
            
            if 'match_info' in result:
                match = result['match_info']
                print(f"✓ Matching exact: {match['match_score']:.1%}")
                print(f"  • Ingrédients en commun: {len(match['matching_ingredients'])}")
                print(f"  • Ingrédients supplémentaires nécessaires: {len(match['additional_needed'])}")
            
            if 'cooking_time' in result:
                print(f"⏱️  Temps: {result['cooking_time']} min")
            
            print(f"\n🥕 Ingrédients ({len(result['ingredients'])}):")
            for ing in result['ingredients'][:10]:  # Afficher max 10 ingrédients
                print(f"   • {ing}")
            if len(result['ingredients']) > 10:
                print(f"   ... et {len(result['ingredients']) - 10} autres")

# ============================================
# INTERFACE CLI SIMPLE
# ============================================

def interactive_recipe_search():
    """Interface en ligne de commande pour rechercher des recettes"""
    print("\n" + "="*70)
    print("🍳 SYSTÈME DE RECOMMANDATION DE RECETTES")
    print("="*70)
    
    # Initialiser le système
    rag = RecipeRAGSystem()
    
    # Charger les données
    print("\n⚙️  Configuration:")
    recipes_path = input(f"Chemin du fichier recettes (défaut: {RECIPES_FOLDER}recipes_prepared_sample.csv): ").strip()
    if not recipes_path:
        recipes_path = f"{RECIPES_FOLDER}recipes_prepared_sample.csv"
    
    rag.load_recipes(recipes_path)
    
    # Charger ou créer les embeddings
    embeddings_path = recipes_path.replace('.csv', '_embeddings.pkl')
    try:
        rag.load_embeddings(embeddings_path)
    except FileNotFoundError:
        print(f"⚠️  Embeddings non trouvés, création en cours...")
        rag.create_embeddings(embeddings_path)
    
    # Boucle de recherche
    while True:
        print("\n" + "="*70)
        ingredients_input = input("\n🥕 Entrez vos ingrédients séparés par des virgules (ou 'quit' pour quitter):\n> ").strip()
        
        exitWord = ['quit', 'exit', 'q']

        if ingredients_input.lower() in exitWord:
            print("\n👋 Au revoir !")
            break

        max_cooking_time = input("⏱️  Temps de cuisson max en minutes (ou Enter pour aucun): ").strip()

        if max_cooking_time.lower() in exitWord:
            print("\n👋 Au revoir !")
            break

        if max_cooking_time.isdigit():
            max_cooking_time = int(max_cooking_time)
        else:
            max_cooking_time = None

        # Parser les ingrédients
        user_ingredients = [ing.strip() for ing in ingredients_input.split(',')]
        
        print(f"\n🔍 Recherche de recettes avec: {user_ingredients}")
        
        # Rechercher
        results = rag.search_with_filters(
            user_ingredients,
            top_k=5,
            min_match_score=0.2
        )
        
        if not results:
            print("\n❌ Aucune recette trouvée. Essayez d'autres ingrédients.")
            continue
        
        # Afficher les résultats
        rag.display_results(results)
        
        # Option pour voir le détail
        detail = input("\n💡 Voir le détail d'une recette ? (numéro ou Enter pour continuer): ").strip()
        if detail.isdigit() and 0 < int(detail) <= len(results):
            recipe = results[int(detail) - 1]
            print("\n" + "="*70)
            print(recipe['recipe_text'])
            print("="*70)


# ============================================
# UTILISATION
# ============================================

if __name__ == "__main__":
    # OPTION 1: Interface interactive (recommandé)
    interactive_recipe_search()
    
    # OPTION 2: Exemple programmatique
    """
    rag = RecipeRAGSystem()
    rag.load_recipes('recipes_sample.csv')
    rag.create_embeddings('recipes_sample_embeddings.pkl')
    
    results = rag.search_with_filters(
        user_ingredients=['chicken', 'garlic', 'lemon'],
        top_k=5
    )
    
    rag.display_results(results)
    """