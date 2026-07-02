from helper_functions import *

food_items = []

def main():
    """Main function for interactive CLI food recommendation system"""
    try:
        print("Food recommendation System")
        print("loading food database...")
        global food_items
        food_items = load_data('./FoodDataSet.json')
        print(f"✅ Loaded {len(food_items)} food items successfully")
        
        # Create and populate search collection
        collection = create_similarity_search_collection(
            'interactive_food_search',
            {'description': ' A collection of interactive food search '}

        )
        populte_similarity_collection(collection, food_items)

        interactive_food_chatbot(collection)
    except Exception as error:
        print("Error initializing system: {error}")
    
def interactive_food_chatbot(collection):
    print("Interactive Food Search Chatbot")
    print("  • Type any food name or description to search")
    print("  • 'help' - Show available commands")
    print("  • 'quit' or 'exit' - Exit the system")
    print("  • Ctrl+C - Emergency exit")

    while True:
        try:
            user_input = input("Search for food: ").strip()

            if not user_input:
                print("Please enter a search term or 'help' for command")
                continue

            if user_input.lower() in ['quit','exit', 'q']:
                print   ("Thankyou for using the food recommendation")
                print("Goodbye")
                break

            elif user_input.lower() in ['help','h']:
                show_help_menu()
            
            else:
                handle_food_search(collection, user_input)

        except KeyboardInterrupt:
            print("System Interuppted. GOODBYE")
            break
        except Exception as e:
            print("Error processing request: {e}")

def show_help_menu():
    """Display help information for users"""
    print("\n📖 HELP MENU")
    print("-" * 30)
    print("Search Examples:")
    print("  • 'chocolate dessert' - Find chocolate desserts")
    print("  • 'Italian food' - Find Italian cuisine")
    print("  • 'sweet treats' - Find sweet desserts")
    print("  • 'baked goods' - Find baked items")
    print("  • 'low calorie' - Find lower-calorie options")
    print("\nCommands:")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the system")

def handle_food_search(collection, query):
    "Handle food similarity search with enhanced display"
    print("Seaarching for '{query}.....")
    print("Please Wait")

    results = perform_similarity_search(collection, query, 5)

    if not results:
        print("❌ No matching foods found.")
        print("💡 Try different keywords like:")
        print("   • Cuisine types: 'Italian', 'American'")
        print("   • Ingredients: 'chocolate', 'flour', 'cheese'")
        print("   • Descriptors: 'sweet', 'baked', 'dessert'")
        return
    
    # Display results 
    print(f"Found {len(results)} recommendations")

    
    for i, result in enumerate(results, 1):
        
        percentage_score = result['similarity_score'] * 100
        print(f"\n{i}. 🍽️  {result['food_name']}")
        print(f"   📊 Match Score: {percentage_score:.1f}%")
        print(f"   🏷️  Cuisine: {result['cuisine_type']}")
        print(f"   🔥 Calories: {result['food_calories_per_serving']} per serving")
        print(f"   📝 Description: {result['food_description']}")
        
        # Add visual separator
        if i < len(results):
            print("   " + "-" * 50)
    

    
    # Provide suggestions for further exploration
    suggest_related_searches(results) 

def suggest_related_searches(results):
    "Suggest related search based on the current result"
    if not results:
        return
    cuisines = list(set(r['cuisine_type'] for r in results))
    print("Related search you might like....")
    for cuisine in cuisines[:3]:
        print(f"   • Try '{cuisine} dishes' for more {cuisine} options")
    
    avg_calories = sum([r['food_calories_per_serving'] for r in results]) / len(results)
    if avg_calories > 350:
        print("   • Try 'low calorie' for lighter options")
    else:
        print("   • Try 'hearty meal' for more substantial dishes")


if __name__ == "__main__":
    main()
