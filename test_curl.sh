#!/bin/bash

# CalorieAI API Test Script using curl
# Make sure your API is running: uvicorn app:app --reload

echo "🧪 Testing CalorieAI API with curl"
echo "=================================="

# Test 1: Meal Suggestion - No meals consumed
echo -e "\n1. Testing meal suggestion with no meals consumed:"
curl -X POST "http://127.0.0.1:8000/suggest-meal" \
  -H "Content-Type: application/json" \
  -d '{
    "todays_meals": [],
    "daily_protein_goal": 120.0
  }' | jq '.'

# Test 2: Meal Suggestion - Some meals consumed
echo -e "\n2. Testing meal suggestion with some meals consumed:"
curl -X POST "http://127.0.0.1:8000/suggest-meal" \
  -H "Content-Type: application/json" \
  -d '{
    "todays_meals": [
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
      }
    ],
    "daily_protein_goal": 120.0
  }' | jq '.'

# Test 3: Transcription Analysis - Single meal
echo -e "\n3. Testing transcription analysis with single meal:"
curl -X POST "http://127.0.0.1:8000/analyze-transcription" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "I had oatmeal with berries for breakfast"
  }' | jq '.'

# Test 4: Transcription Analysis - Multiple meals
echo -e "\n4. Testing transcription analysis with multiple meals:"
curl -X POST "http://127.0.0.1:8000/analyze-transcription" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "For lunch I had grilled chicken with rice, then I ate a Greek yogurt for snack"
  }' | jq '.'

# Test 5: Image Analysis (requires real image URL)
echo -e "\n5. Testing image analysis:"
curl -X POST "http://127.0.0.1:8000/analyze-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=500&h=500&fit=crop"
  }' | jq '.'

echo -e "\n✅ Testing completed!"
echo -e "\nNote: Make sure you have jq installed for pretty JSON output: brew install jq"
echo -e "If you don't have jq, remove the '| jq .' from the commands above"
