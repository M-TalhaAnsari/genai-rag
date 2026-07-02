from helper_functions import *
from typing import List, Dict, Any
import ollama
import json

# Global variables
food_items = []
OLLAMA_MODEL = "llama3.2"
def generate_response(prompt: str) -> str:
    """Generate response using Ollama"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def main():
    """Main function for enhanced RAG chatbot system"""
    try:
        print("🤖 Enhanced RAG-Powered Food Recommendation Chatbot")
    
        # Load food data
        global food_items
        food_items = load_data('./FoodDataSet.json')
        print(f"✅ Loaded {len(food_items)} food items")
        
        # Create collection for RAG system
        collection = create_similarity_search_collection(
            "enhanced_rag_food_chatbot",
            {'description': 'Enhanced RAG chatbot with IBM watsonx.ai integration'}
        )
        populate_similarity_collection(collection, food_items)
        print("✅ Vector database ready")
        
        # Test LLM connection
        print(" Testing LLM connection...")
        try:
            response = generate_response("Say hello.")

            if response:
                print("✅ Ollama connected")
            else:
                print("❌ Ollama returned empty response")
                return

        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            return
        
        # Start enhanced RAG chatbot
        enhanced_rag_food_chatbot(collection)
        
    except Exception as error:
        print(f" Error: {error}")

def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Prepare structured context from search results for LLM"""
    if not search_results:
        return "No relevant food items found in the database."
    
    context_parts = []
    context_parts.append("Based on your query, here are the most relevant food options from our database:")
    context_parts.append("")
    
    for i, result in enumerate(search_results[:3], 1):
        food_context = []
        food_context.append(f"Option {i}: {result['food_name']}")
        food_context.append(f"  - Description: {result['food_description']}")
        food_context.append(f"  - Cuisine: {result['cuisine_type']}")
        food_context.append(f"  - Calories: {result['food_calories_per_serving']} per serving")
        
        if result.get('food_ingredients'):
            ingredients = result['food_ingredients']
            if isinstance(ingredients, list):
                food_context.append(f"  - Key ingredients: {', '.join(ingredients[:5])}")
            else:
                food_context.append(f"  - Key ingredients: {ingredients}")
        
        if result.get('food_health_benefits'):
            food_context.append(f"  - Health benefits: {result['food_health_benefits']}")
        
        if result.get('cooking_method'):
            food_context.append(f"  - Cooking method: {result['cooking_method']}")
        
        if result.get('taste_profile'):
            food_context.append(f"  - Taste profile: {result['taste_profile']}")
        
        food_context.append(f"  - Similarity score: {result['similarity_score']*100:.1f}%")
        food_context.append("")
        
        context_parts.extend(food_context)
    
    return "\n".join(context_parts)

def generate_llm_rag_response(query: str, search_results: List[Dict]) -> str:
    """Generate response using IBM Granite with retrieved context"""
    
        # Prepare context from search results
    context = prepare_context_for_llm(query, search_results)        
        # Build the prompt for the LLM
    prompt = f'''You are a helpful food recommendation assistant. A user is asking for food recommendations, and I've retrieved relevant options from a food database.
User Query: "{query}"

Retrieved Food Information:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Recommends 2-3 specific food items from the retrieved options
3. Explains why these recommendations match their request
4. Includes relevant details like cuisine type, calories, or health benefits
5. Uses a friendly, conversational tone
6. Keeps the response concise but informative

Response:'''

    try:
        generated_response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = generated_response["message"]["content"].strip()

        if len(response_text) < 50:
            return generate_fallback_response(query, search_results)

        return response_text

    except Exception as e:
        print(f"LLM Error: {e}")
        return generate_fallback_response(query, search_results)

def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate fallback response when LLM fails"""
    if not search_results:
        return "I couldn't find any food items matching your request. Try describing what you're in the mood for with different words!"
    
    top_result = search_results[0]
    response_parts = []
    
    response_parts.append(f"Based on your request for '{query}', I'd recommend {top_result['food_name']}.")
    response_parts.append(f"It's a {top_result['cuisine_type']} dish with {top_result['food_calories_per_serving']} calories per serving.")
    
    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(f"Another great option would be {second_choice['food_name']}.")
    
    return " ".join(response_parts)

def enhanced_rag_food_chatbot(collection):
    """Enhanced RAG-powered conversational food chatbot with IBM Granite"""
  
    print("ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print("   Powered by IBM's Granite Model")

    print("Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("  • 'I'm craving comfort food for a cold evening'")
    print("  • 'Suggest some protein-rich breakfast options'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\ You: ").strip()
            
            if not user_input:
                print(" Bot: Please tell me what kind of food you're looking for!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Bot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Hope you found some delicious recommendations! 👋")
                break
            
            elif user_input.lower() in ['help', 'h']:
                show_enhanced_rag_help()
            
            elif user_input.lower() in ['compare']:
                handle_enhanced_comparison_mode(collection)
            
            else:
                # Process the food query with enhanced RAG
                handle_enhanced_rag_query(collection, user_input, conversation_history)
                conversation_history.append(user_input)
                
                # Keep conversation history manageable
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-3:]
                
        except KeyboardInterrupt:
            print("\n\n Bot: Goodbye! Hope you find something delicious! 👋")
            break
        except Exception as e:
            print(f" Bot: Sorry, I encountered an error: {e}")

def handle_enhanced_rag_query(collection, query: str, conversation_history: List[str]):
    """Handle user query with enhanced RAG approach using IBM Granite"""
    print(f"\n🔍 Searching vector database for: '{query}'...")
    
    # Perform similarity search with more results for better context
    search_results = perform_similarity_search(collection, query, 3)
    
    if not search_results:
        print(" Bot: I couldn't find any food items matching your request.")
        print("      Try describing what you're in the mood for with different words!")
        return
    
    print(f" Found {len(search_results)} relevant matches")
    print("Generating AI-powered response...")
    
    # Generate enhanced RAG response using IBM Granite
    ai_response = generate_llm_rag_response(query, search_results)
    
    print(f" Bot: {ai_response}")
    
    # Show detailed results for reference
    print(f" Search Results Details:")

    for i, result in enumerate(search_results[:3], 1):
        print(f"{i}.   {result['food_name']}")
        print(f"    {result['cuisine_type']} |  {result['food_calories_per_serving']} cal |  {result['similarity_score']*100:.1f}% match")
        if i < 3:
            print()

def handle_enhanced_comparison_mode(collection):
    """Enhanced comparison between two food queries using LLM"""
    print("\n ENHANCED COMPARISON MODE")
    print("   Powered by AI Analysis")
    
    query1 = input("Enter first food query: ").strip()
    query2 = input("Enter second food query: ").strip()
    
    if not query1 or not query2:
        print("Please enter both queries for comparison")
        return
    
    print(f"\n Analyzing '{query1}' vs '{query2}' with AI...")
    
    # Get results for both queries
    results1 = perform_similarity_search(collection, query1, 3)
    results2 = perform_similarity_search(collection, query2, 3)
    
    # Generate AI-powered comparison
    comparison_response = generate_llm_comparison(query1, query2, results1, results2)
    
    print(f"\n🤖 AI Analysis: {comparison_response}")
    
    # Show side-by-side results
    print(f"\n DETAILED COMPARISON")

    print(f"{'Query 1: ' + query1[:20] + '...' if len(query1) > 20 else 'Query 1: ' + query1:<30} | {'Query 2: ' + query2[:20] + '...' if len(query2) > 20 else 'Query 2: ' + query2}")

    
    max_results = max(len(results1), len(results2))
    for i in range(min(max_results, 3)):
        left = f"{results1[i]['food_name']} ({results1[i]['similarity_score']*100:.0f}%)" if i < len(results1) else "---"
        right = f"{results2[i]['food_name']} ({results2[i]['similarity_score']*100:.0f}%)" if i < len(results2) else "---"
        print(f"{left[:30]:>30} | {right[:30]}")

def generate_llm_comparison(query1: str, query2: str,
                            results1: List[Dict],
                            results2: List[Dict]) -> str:
    """Generate AI-powered comparison between two queries"""

    try:
        context1 = prepare_context_for_llm(query1, results1[:3])
        context2 = prepare_context_for_llm(query2, results2[:3])

        comparison_prompt = f"""You are analyzing and comparing two different food preference queries.

Query 1: "{query1}"
Top Results:
{context1}

Query 2: "{query2}"
Top Results:
{context2}

Please:
1. Compare both queries.
2. Mention similarities and differences.
3. Recommend the best option from each.
4. Keep the answer concise.
"""

        generated_response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": comparison_prompt
                }
            ]
        )

        return generated_response["message"]["content"].strip()

    except Exception as e:
        print(f"LLM Error: {e}")
        return generate_simple_comparison(query1, query2, results1, results2)
    
def generate_simple_comparison(query1: str, query2: str, results1: List[Dict], results2: List[Dict]) -> str:
    """Simple comparison fallback"""
    if not results1 and not results2:
        return "No results found for either query."
    if not results1:
        return f"Found results for '{query2}' but none for '{query1}'."
    if not results2:
        return f"Found results for '{query1}' but none for '{query2}'."
    
    return f"For '{query1}', I recommend {results1[0]['food_name']}. For '{query2}', {results2[0]['food_name']} would be perfect."

def show_enhanced_rag_help():
    """Display help information for enhanced RAG chatbot"""
    print("\n ENHANCED RAG CHATBOT HELP")

    print(" This chatbot uses IBM watsonx.ai to understand your")
    print("   food preferences and provide intelligent recommendations.")
    print("\nHow to get the best recommendations:")
    print("  • Be specific: 'healthy Italian pasta under 350 calories'")
    print("  • Mention preferences: 'spicy comfort food for cold weather'")
    print("  • Include context: 'light breakfast for busy morning'")
    print("  • Ask about benefits: 'protein-rich foods for workout recovery'")
    print("\nSpecial features:")
    print("  •  Vector similarity search finds relevant foods")
    print("  •  AI analysis provides contextual explanations")
    print("  •  Detailed nutritional and cuisine information")
    print("  •  Smart comparison between different preferences")
    print("\nCommands:")
    print("  • 'compare' - AI-powered comparison of two queries")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the chatbot")
    print("\nTips for better results:")
    print("  • Use natural language - talk like you would to a friend")
    print("  • Mention dietary restrictions or preferences")
    print("  • Include meal timing (breakfast, lunch, dinner)")
    print("  • Specify if you want healthy, comfort, or indulgent options")

if __name__ == "__main__":
    main()
