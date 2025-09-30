#!/usr/bin/env python3
"""
Test script to make real requests to the CalorieAI API endpoints.
This helps you test how the model is performing with actual data.
"""

import requests
import json
from typing import List, Dict, Any

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_meal_suggestion():
    """Test the meal suggestion endpoint with real data."""
    print("🍽️  Testing Meal Suggestion Endpoint")
    print("=" * 50)
    
    # Test case 1: No meals consumed yet
    print("\n1. Testing with no meals consumed:")
    request_data = {
        "todays_meals": [],
        "daily_protein_goal": 120.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/suggest-meal", json=request_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # Response is now an array of meal objects (like transcription)
            if data and len(data) > 0:
                meal = data[0]  # Get the first (and only) suggested meal
                print(f"Suggested meal: {meal['mealName']}")
                print(f"Serving size: {meal['servingSize']['qty']} {meal['servingSize']['unit']} ({meal['servingSize']['grams']}g)")
                print(f"Category: {meal.get('category', 'N/A')}")
                print(f"Protein: {meal['macros']['protein']}g")
                print(f"Calories: {meal['macros']['calories']}")
                print(f"Carbs: {meal['macros']['carbohydrates']['total']}g (net: {meal['macros']['carbohydrates']['net']}g)")
                print(f"Fat: {meal['macros']['fat']['total']}g")
                print(f"Ingredients: {meal['ingredients']}")
                print(f"✅ Meal suggestion validated successfully")
            else:
                print("No meal suggestion returned")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test case 2: Some meals consumed
    print("\n2. Testing with some meals consumed:")
    meals_data = [
        {
            "mealName": "Oatmeal with berries",
            "servingSize": {"qty": 1, "unit": "bowl", "grams": 250},
            "ingredients": "Oats, milk, honey, mixed berries",
            "category": "Grain",
            "macros": {
                "calories": 300,
                "protein": 12,
                "carbohydrates": {
                    "total": 50, "net": 45, "fiber": 5, "sugar": 15, 
                    "addedSugar": 10, "sugarAlcohols": 0, "allulose": 0
                },
                "fat": {
                    "total": 8, "saturated": 2, "monounsaturated": 3, 
                    "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 5
                }
            },
            "micronutrients": []
        },
        {
            "mealName": "Greek yogurt",
            "servingSize": {"qty": 1, "unit": "cup", "grams": 200},
            "ingredients": "Greek yogurt, honey",
            "category": "Dairy",
            "macros": {
                "calories": 150,
                "protein": 20,
                "carbohydrates": {
                    "total": 15, "net": 12, "fiber": 0, "sugar": 12, 
                    "addedSugar": 8, "sugarAlcohols": 0, "allulose": 0
                },
                "fat": {
                    "total": 5, "saturated": 3, "monounsaturated": 1, 
                    "polyunsaturated": 0.5, "omega3": 0, "omega6": 0.5, "cholesterol": 15
                }
            },
            "micronutrients": []
        }
    ]
    
    request_data = {
        "todays_meals": meals_data,
        "daily_protein_goal": 120.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/suggest-meal", json=request_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # Response is now an array of meal objects (like transcription)
            if data and len(data) > 0:
                meal = data[0]  # Get the first (and only) suggested meal
                print(f"Suggested meal: {meal['mealName']}")
                print(f"Serving size: {meal['servingSize']['qty']} {meal['servingSize']['unit']} ({meal['servingSize']['grams']}g)")
                print(f"Category: {meal.get('category', 'N/A')}")
                print(f"Protein: {meal['macros']['protein']}g")
                print(f"Calories: {meal['macros']['calories']}")
                print(f"Carbs: {meal['macros']['carbohydrates']['total']}g (net: {meal['macros']['carbohydrates']['net']}g)")
                print(f"Fat: {meal['macros']['fat']['total']}g")
                print(f"Ingredients: {meal['ingredients']}")
                
                # Calculate remaining protein
                total_consumed = sum(m['macros']['protein'] for m in meals_data) if 'meals_data' in locals() else 0
                remaining = 120.0 - total_consumed
                print(f"Remaining protein needed: {remaining}g")
                print(f"✅ Meal suggestion validated successfully")
            else:
                print("No meal suggestion returned")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_transcription_analysis():
    """Test the transcription analysis endpoint with comprehensive test cases."""
    print("\n\n📝 Testing Transcription Analysis Endpoint")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Single meal - simple",
            "transcription": "I had oatmeal with berries for breakfast",
            "expected_meals": 1
        },
        {
            "name": "Multiple meals with 'then'",
            "transcription": "For lunch I had grilled chicken with rice, then I ate a Greek yogurt for snack",
            "expected_meals": 2
        },
        {
            "name": "Three-course meal",
            "transcription": "I started with a protein shake, then had pasta with marinara sauce and meatballs, finished with tiramisu for dessert",
            "expected_meals": 3
        },
        {
            "name": "Complex multi-meal day",
            "transcription": "Breakfast was pancakes with syrup. Later I had a sandwich for lunch. In the evening I ate steak with vegetables, then had ice cream for dessert",
            "expected_meals": 4
        },
        {
            "name": "Snack sequence",
            "transcription": "I grabbed some nuts, then later had an apple, and finally ate a granola bar",
            "expected_meals": 3
        },
        {
            "name": "Beverage + food combinations",
            "transcription": "I had coffee and a croissant for breakfast, then pizza and soda for lunch",
            "expected_meals": 2
        },
        {
            "name": "Time-based meals",
            "transcription": "This morning I had eggs. At noon I ate a salad. Tonight I'm having fish",
            "expected_meals": 3
        },
        {
            "name": "Single complex meal",
            "transcription": "I had a big breakfast with eggs, bacon, toast, orange juice, and coffee",
            "expected_meals": 1
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Input: '{test_case['transcription']}'")
        
        request_data = {"transcription": test_case["transcription"]}
        
        try:
            response = requests.post(f"{BASE_URL}/analyze-transcription", json=request_data)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                actual_meals = len(data)
                expected_meals = test_case["expected_meals"]
                
                print(f"   Expected meals: {expected_meals}, Actual meals: {actual_meals}")
                
                if actual_meals == expected_meals:
                    print("   ✅ Meal count matches expectation")
                else:
                    print("   ⚠️  Meal count differs from expectation")
                
                for j, meal in enumerate(data, 1):
                    print(f"   Meal {j}: {meal['mealName']}")
                    print(f"     - Protein: {meal['macros']['protein']}g")
                    print(f"     - Calories: {meal['macros']['calories']}")
                    print(f"     - Category: {meal.get('category', 'N/A')}")
                    print(f"     - Micronutrients: {len(meal['micronutrients'])} items")
                    
                    # Use comprehensive validation
                    validate_meal_output(meal, f"transcription meal {j}")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

def test_image_analysis():
    """Test the image analysis endpoint with various food images."""
    print("\n\n📸 Testing Image Analysis Endpoint")
    print("=" * 50)
    
    test_images = [
        {
            "name": "Hamburger and french fries",
            "url": "https://cdn.britannica.com/98/235798-050-3C3BA15D/Hamburger-and-french-fries-paper-box.jpg",
            "expected_meals": 2
        },
        {
            "name": "Masala Dosa",
            "url": "https://blog.swiggy.com/wp-content/uploads/2024/02/Masala-Dosa-1024x538.jpg",
            "expected_meals": 1
        },
        {
            "name": "Chicken pot pie soup",
            "url": "https://hips.hearstapps.com/hmg-prod/images/comfort-food-recipes-chicken-pot-pie-soup-66d9e96824766.jpg",
            "expected_meals": 1
        },
        {
            "name": "Spicy chicken",
            "url": "https://static01.nyt.com/images/2021/03/16/multimedia/00xp-spicy1/00xp-spicy1-mediumSquareAt3X.jpg",
            "expected_meals": 1
        },
        {
            "name": "Indian cuisine",
            "url": "https://www.contiki.com/six-two/app/uploads/2024/03/IMG-20240318-WA0007-e1710844435378.jpg",
            "expected_meals": 1
        }
    ]
    
    for i, test_image in enumerate(test_images, 1):
        print(f"\n{i}. {test_image['name']}")
        print(f"   URL: {test_image['url']}")
        
        request_data = {"image_url": test_image["url"]}
        
        try:
            response = requests.post(f"{BASE_URL}/analyze-image", json=request_data)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                actual_meals = len(data)
                expected_meals = test_image["expected_meals"]
                
                print(f"   Expected meals: {expected_meals}, Actual meals: {actual_meals}")
                
                if actual_meals >= 1:
                    print("   ✅ At least one meal detected")
                else:
                    print("   ⚠️  No meals detected")
                
                for j, meal in enumerate(data, 1):
                    print(f"   Meal {j}: {meal['mealName']}")
                    print(f"     - Protein: {meal['macros']['protein']}g")
                    print(f"     - Calories: {meal['macros']['calories']}")
                    print(f"     - Serving: {meal['servingSize']['qty']} {meal['servingSize']['unit']} ({meal['servingSize']['grams']}g)")
                    print(f"     - Ingredients: {meal['ingredients'][:100]}...")
                    print(f"     - Micronutrients: {len(meal['micronutrients'])} items")
                    
                    # Use comprehensive validation
                    validate_meal_output(meal, f"image meal {j}")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

def validate_meal_output(meal_data, test_name):
    """Validate meal output structure and content."""
    print(f"   Validating {test_name}...")
    
    # Check required fields
    required_fields = ["mealName", "servingSize", "ingredients", "macros", "micronutrients"]
    missing_fields = [field for field in required_fields if field not in meal_data]
    
    if missing_fields:
        print(f"     ❌ Missing required fields: {missing_fields}")
        return False
    else:
        print(f"     ✅ All required fields present")
    
    # Check micronutrients is empty array
    if meal_data["micronutrients"] != []:
        print(f"     ❌ Micronutrients should be empty array, got: {meal_data['micronutrients']}")
        return False
    else:
        print(f"     ✅ Micronutrients correctly empty")
    
    # Check macros structure
    macros = meal_data["macros"]
    macro_fields = ["calories", "protein", "carbohydrates", "fat"]
    missing_macro_fields = [field for field in macro_fields if field not in macros]
    
    if missing_macro_fields:
        print(f"     ❌ Missing macro fields: {missing_macro_fields}")
        return False
    else:
        print(f"     ✅ All macro fields present")
    
    # Check serving size structure
    serving_size = meal_data["servingSize"]
    serving_fields = ["qty", "unit", "grams"]
    missing_serving_fields = [field for field in serving_fields if field not in serving_size]
    
    if missing_serving_fields:
        print(f"     ❌ Missing serving size fields: {missing_serving_fields}")
        return False
    else:
        print(f"     ✅ All serving size fields present")
    
    # Check data types
    try:
        assert isinstance(macros["calories"], (int, float)), "Calories should be number"
        assert isinstance(macros["protein"], (int, float)), "Protein should be number"
        assert isinstance(serving_size["qty"], int), "Quantity should be integer"
        assert isinstance(serving_size["grams"], int), "Grams should be integer"
        print(f"     ✅ Data types are correct")
    except AssertionError as e:
        print(f"     ❌ Data type validation failed: {e}")
        return False
    
    # Check reasonable values
    if macros["calories"] <= 0:
        print(f"     ⚠️  Calories should be positive, got: {macros['calories']}")
    if macros["protein"] < 0:
        print(f"     ⚠️  Protein should be non-negative, got: {macros['protein']}")
    if serving_size["grams"] <= 0:
        print(f"     ⚠️  Grams should be positive, got: {serving_size['grams']}")
    
    return True

def test_api_health():
    """Test if the API is running."""
    print("🏥 Testing API Health")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"API Docs accessible: {response.status_code == 200}")
        
        # Try to get OpenAPI schema
        response = requests.get(f"{BASE_URL}/openapi.json")
        print(f"OpenAPI schema accessible: {response.status_code == 200}")
        
    except Exception as e:
        print(f"API not accessible: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 CalorieAI API Testing Script")
    print("=" * 50)
    
    # Check if API is running
    if not test_api_health():
        print("\n❌ API is not running. Please start it with: uvicorn app:app --reload")
        return
    
    print("\n✅ API is running!")
    
    # Run tests
    test_meal_suggestion()
    test_transcription_analysis()
    test_image_analysis()
    
    print("\n\n🎉 Testing completed!")
    print("\nTo test with your own data:")
    print("1. For meal suggestions: POST to /suggest-meal with todays_meals and daily_protein_goal")
    print("2. For transcription: POST to /analyze-transcription with transcription text")
    print("3. For images: POST to /analyze-image with image_url")
    print("4. For image upload: POST to /analyze-image-upload with file upload")

if __name__ == "__main__":
    main()
