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
            print(f"Suggested meal: {data['suggested_meal']}")
            print(f"Estimated protein: {data['estimated_protein']}g")
            print(f"Estimated calories: {data['estimated_calories']}")
            print(f"Reason: {data['reason']}")
            print(f"Remaining protein needed: {data['remaining_protein_needed']}g")
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
            "micronutrients": [
                {"name": "Protein", "amount": 12, "unit": "g"},
                {"name": "Fiber", "amount": 5, "unit": "g"}
            ]
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
            "micronutrients": [
                {"name": "Protein", "amount": 20, "unit": "g"},
                {"name": "Calcium", "amount": 200, "unit": "mg"}
            ]
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
            print(f"Suggested meal: {data['suggested_meal']}")
            print(f"Estimated protein: {data['estimated_protein']}g")
            print(f"Estimated calories: {data['estimated_calories']}")
            print(f"Reason: {data['reason']}")
            print(f"Remaining protein needed: {data['remaining_protein_needed']}g")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_transcription_analysis():
    """Test the transcription analysis endpoint."""
    print("\n\n📝 Testing Transcription Analysis Endpoint")
    print("=" * 50)
    
    test_transcriptions = [
        "I had oatmeal with berries for breakfast",
        "For lunch I had grilled chicken with rice, then I ate a Greek yogurt for snack",
        "I started with a protein shake, then had pasta with marinara sauce and meatballs, finished with tiramisu for dessert"
    ]
    
    for i, transcription in enumerate(test_transcriptions, 1):
        print(f"\n{i}. Testing transcription: '{transcription}'")
        
        request_data = {"transcription": transcription}
        
        try:
            response = requests.post(f"{BASE_URL}/analyze-transcription", json=request_data)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Number of meals detected: {len(data)}")
                for j, meal in enumerate(data, 1):
                    print(f"  Meal {j}: {meal['mealName']} - {meal['macros']['protein']}g protein, {meal['macros']['calories']} calories")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

def test_image_analysis():
    """Test the image analysis endpoint (requires a real image URL)."""
    print("\n\n📸 Testing Image Analysis Endpoint")
    print("=" * 50)
    
    # You can replace this with a real food image URL
    test_image_url = "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=500&h=500&fit=crop"
    
    print(f"Testing with image URL: {test_image_url}")
    
    request_data = {"image_url": test_image_url}
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-image", json=request_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Number of meals detected: {len(data)}")
            for i, meal in enumerate(data, 1):
                print(f"  Meal {i}: {meal['mealName']} - {meal['macros']['protein']}g protein, {meal['macros']['calories']} calories")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

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
