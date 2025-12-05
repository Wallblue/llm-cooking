import requests
import json
from typing import List, Dict, Optional
from phase2_rag import RecipeRAGSystem

# ============================================
# PHASE 3 : CHAT CONVERSATIONNEL AVEC LLM LOCAL
# ============================================

class RecipeChatBot:
    """
    Chat bot conversationnel pour recommandations de recettes
    Utilise Ollama (Llama 3.2) + RAG pour des réponses intelligentes
    """
    
    def __init__(self, model_name='llama3.2', ollama_url='http://localhost:11434'):
        """
        Initialise le chat bot
        
        Args:
            model_name: Modèle Ollama à utiliser (llama3.2, llama2, mistral, etc.)
            ollama_url: URL de l'API Ollama
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.conversation_history = []
        
        # Initialiser le système RAG (Phase 2)
        print("🤖 Initialisation du chat bot...")
        self.rag = RecipeRAGSystem()
        
        # Vérifier qu'Ollama est disponible
        if not self._check_ollama():
            raise ConnectionError(
                "❌ Ollama n'est pas accessible. "
                "Assurez-vous qu'Ollama est lancé avec: ollama serve"
            )
        
        print(f"✓ Chat bot initialisé avec le modèle: {model_name}")
    
    def _check_ollama(self) -> bool:
        """Vérifie qu'Ollama est accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def load_recipes(self, recipes_path: str, embeddings_path: Optional[str] = None):
        """Charge les recettes et leurs embeddings via le système RAG"""
        self.rag.load_recipes(recipes_path)
        
        if embeddings_path:
            try:
                self.rag.load_embeddings(embeddings_path)
            except FileNotFoundError:
                print("⚠️  Embeddings non trouvés, création en cours...")
                self.rag.create_embeddings(embeddings_path)
        else:
            # Créer un nom d'embeddings automatique
            embeddings_path = recipes_path.replace('.csv', '_embeddings.pkl')
            try:
                self.rag.load_embeddings(embeddings_path)
            except FileNotFoundError:
                self.rag.create_embeddings(embeddings_path)
        
        print("✓ Système prêt !")
    
    def _detect_ingredients(self, user_message: str) -> Optional[List[str]]:
        """
        Détecte si le message contient des ingrédients
        Utilise des heuristiques simples
        """
        # Mots-clés indiquant une recherche d'ingrédients
        keywords = [
            "j'ai", "avec", "ingrédients", "utiliser", 
            "cuisiner", "préparer", "faire avec", "recette",
            "i have", "using", "cook with", "make with"
        ]
        
        # Vérifier si le message contient ces mots-clés
        message_lower = user_message.lower()
        has_keyword = any(kw in message_lower for kw in keywords)
        
        if not has_keyword:
            return None
        
        # Extraire les mots qui pourraient être des ingrédients
        # (heuristique simple : mots de 3+ lettres)
        words = user_message.replace(',', ' ').split()
        potential_ingredients = [
            word.strip('.,!?;:')
            for word in words
            if len(word) >= 3 and word.lower() not in keywords
        ]
        
        return potential_ingredients if potential_ingredients else None
    
    def _format_recipes_for_prompt(self, recipes: List[Dict]) -> str:
        """Formate les recettes pour le prompt du LLM"""
        if not recipes:
            return "Aucune recette trouvée."
        
        formatted = "Voici les meilleures recettes que j'ai trouvées :\n\n"
        
        for i, recipe in enumerate(recipes, 1):
            formatted += f"### Recette {i}: {recipe['name']}\n"
            formatted += f"Score de pertinence: {recipe['similarity_score']:.1%}\n"
            
            if 'cooking_time' in recipe:
                formatted += f"Temps de préparation: {recipe['cooking_time']} minutes\n"
            
            if 'match_info' in recipe:
                match = recipe['match_info']
                formatted += f"Ingrédients en commun: {len(match['matching_ingredients'])}\n"
                formatted += f"Ingrédients supplémentaires nécessaires: {len(match['additional_needed'])}\n"
            
            formatted += f"\nIngrédients ({len(recipe['ingredients'])}):\n"
            for ing in recipe['ingredients'][:10]:  # Limiter pour le prompt
                formatted += f"- {ing}\n"
            if len(recipe['ingredients']) > 10:
                formatted += f"- ... et {len(recipe['ingredients']) - 10} autres\n"
            
            if recipe.get('description'):
                formatted += f"\nDescription: {recipe['description']}\n"
            
            # Ajouter quelques étapes si disponibles
            if recipe.get('steps'):
                steps = recipe['steps']
                if isinstance(steps, str):
                    try:
                        steps = eval(steps)
                    except:
                        steps = [steps]
                
                if steps and len(steps) > 0:
                    formatted += f"\nPremières étapes:\n"
                    for j, step in enumerate(steps[:3], 1):
                        formatted += f"{j}. {step}\n"
                    if len(steps) > 3:
                        formatted += f"... et {len(steps) - 3} autres étapes\n"
            
            formatted += "\n" + "-"*50 + "\n\n"
        
        return formatted
    
    def _create_system_prompt(self) -> str:
        """Crée le prompt système pour définir le comportement du bot"""
        return """Tu es Chef AI, un assistant culinaire expert, sympathique et passionné.

Ton rôle:
- Aider les utilisateurs à trouver et préparer des recettes
- Répondre de manière conversationnelle et chaleureuse
- Donner des conseils culinaires pratiques
- Adapter tes réponses au niveau de l'utilisateur

Style de communication:
- Utilise des emojis avec modération (🍳, 👨‍🍳, 🥕, etc.)
- Sois enthousiaste mais pas excessif
- Pose des questions pour mieux comprendre les besoins
- Donne des explications claires et structurées

Contraintes:
- Base-toi UNIQUEMENT sur les recettes fournies dans le contexte
- Si tu ne trouves pas de recette pertinente, propose des alternatives
- Ne pas inventer d'informations sur les recettes
- Reste concentré sur la cuisine et les recettes"""
    
    def _call_ollama(self, user_message: str, context: str = "") -> str:
        """
        Appelle l'API Ollama pour générer une réponse
        
        Args:
            user_message: Message de l'utilisateur
            context: Contexte additionnel (recettes trouvées, etc.)
        
        Returns:
            Réponse générée par le LLM
        """
        # Construire le prompt complet
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt()
            }
        ]
        
        # Ajouter l'historique de conversation (5 derniers messages max)
        for msg in self.conversation_history[-5:]:
            messages.append(msg)
        
        # Ajouter le contexte si présent
        if context:
            messages.append({
                "role": "system",
                "content": f"Contexte des recettes:\n{context}"
            })
        
        # Ajouter le nouveau message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Appeler Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Créativité modérée
                        "top_p": 0.9,
                    }
                },
                timeout=120  # 2 minutes max
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content']
            else:
                return f"❌ Erreur Ollama (code {response.status_code}): {response.text}"
        
        except requests.exceptions.Timeout:
            return "⏱️ Le modèle met trop de temps à répondre. Essayez une question plus simple."
        except Exception as e:
            return f"❌ Erreur lors de l'appel à Ollama: {str(e)}"
    
    def chat(self, user_message: str) -> str:
        """
        Méthode principale pour interagir avec le chat bot
        
        Args:
            user_message: Message de l'utilisateur
        
        Returns:
            Réponse du bot
        """
        # Détecter si le message contient des ingrédients
        ingredients = self._detect_ingredients(user_message)
        
        context = ""
        if ingredients:
            # Rechercher des recettes avec le système RAG (Phase 2)
            print(f"🔍 Recherche de recettes avec: {ingredients[:5]}...")
            recipes = self.rag.search_with_filters(
                user_ingredients=ingredients,
                top_k=3,
                min_match_score=0.2
            )
            
            if recipes:
                context = self._format_recipes_for_prompt(recipes)
                print(f"✓ {len(recipes)} recettes trouvées")
        
        # Générer la réponse avec le LLM
        print("🤖 Génération de la réponse...")
        bot_response = self._call_ollama(user_message, context)
        
        # Sauvegarder dans l'historique
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": bot_response
        })
        
        return bot_response
    
    def clear_history(self):
        """Efface l'historique de conversation"""
        self.conversation_history = []
        print("🗑️  Historique effacé")
    
    def get_history_summary(self) -> str:
        """Retourne un résumé de l'historique"""
        if not self.conversation_history:
            return "Aucun historique"
        
        summary = f"📜 Historique ({len(self.conversation_history)} messages):\n\n"
        for msg in self.conversation_history[-10:]:  # 10 derniers messages
            role = "👤 Vous" if msg["role"] == "user" else "🤖 Chef AI"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary += f"{role}: {content}\n\n"
        
        return summary


# ============================================
# INTERFACE CLI INTERACTIVE
# ============================================

def interactive_chat():
    """Interface en ligne de commande pour le chat"""
    print("\n" + "="*70)
    print("👨‍🍳 CHEF AI - ASSISTANT CULINAIRE CONVERSATIONNEL")
    print("="*70)
    print("\nPowered by Ollama (Llama 3.2) + RAG\n")
    
    # Initialiser le bot
    try:
        bot = RecipeChatBot(model_name='llama3.2')
    except ConnectionError as e:
        print(f"\n{e}")
        print("\n💡 Pour démarrer Ollama:")
        print("   1. Ouvrez un terminal")
        print("   2. Lancez: ollama serve")
        print("   3. Relancez ce script\n")
        return
    
    # Charger les recettes
    recipes_path = input("Chemin du fichier recettes (défaut: recipes/recipes_prepared.csv): ").strip()
    if not recipes_path:
        recipes_path = "recipes/recipes_prepared.csv"
    
    try:
        bot.load_recipes(recipes_path)
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {recipes_path}")
        return
    
    # Message de bienvenue
    print("\n" + "="*70)
    print("✅ Chef AI est prêt à vous aider !")
    print("="*70)
    print("\n💡 Commandes spéciales:")
    print("   • 'clear' - Efface l'historique de conversation")
    print("   • 'history' - Affiche l'historique")
    print("   • 'quit' | 'q' | 'exit' - Quitter")
    print("\n📝 Exemples de questions:")
    print("   • J'ai du poulet, de l'ail et du citron, que puis-je faire ?")
    print("   • Propose-moi une recette végétarienne rapide")
    print("   • Comment faire cuire un steak parfaitement ?")
    print("   • J'ai 30 minutes, qu'est-ce que tu recommandes ?")
    
    # Boucle de chat
    while True:
        print("\n" + "─"*70)
        user_input = input("\n👤 Vous: ").strip()
        
        if not user_input:
            continue
        
        # Commandes spéciales
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 À bientôt ! Bon appétit !")
            break
        
        if user_input.lower() == 'clear':
            bot.clear_history()
            continue
        
        if user_input.lower() == 'history':
            print("\n" + bot.get_history_summary())
            continue
        
        # Obtenir la réponse du bot
        print("\n🤖 Chef AI: ", end="", flush=True)
        response = bot.chat(user_input)
        print(response)


# ============================================
# UTILISATION
# ============================================

if __name__ == "__main__":
    # Lancer l'interface interactive
    interactive_chat()
    
    # OU utilisation programmatique:
    """
    bot = RecipeChatBot(model_name='llama3.2')
    bot.load_recipes('recipes/recipes_prepared_sample.csv')
    
    response = bot.chat("J'ai du poulet, de l'ail et du citron")
    print(response)
    
    response = bot.chat("Donne-moi plus de détails sur la première recette")
    print(response)
    
    bot.clear_history()
    """