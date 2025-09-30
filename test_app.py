import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app, MealTRANSCRIPTION, ServingSize, Macronutrients, Carbohydrates, Fat, Micronutrient

# Create test client
client = TestClient(app)

# Test data
def create_sample_meal(meal_name: str, protein: float, calories: float) -> MealTRANSCRIPTION:
    """Helper function to create sample meal data for testing."""
    return MealTRANSCRIPTION(
        mealName=meal_name,
        servingSize=ServingSize(qty=1, unit="plate", grams=300),
        ingredients="Sample ingredients",
        category="Test",
        macros=Macronutrients(
            calories=calories,
            protein=protein,
            carbohydrates=Carbohydrates(
                total=30, net=25, fiber=5, sugar=5, addedSugar=0, sugarAlcohols=0, allulose=0
            ),
            fat=Fat(
                total=15, saturated=3, monounsaturated=6, polyunsaturated=3, omega3=1, omega6=2, cholesterol=20
            )
        ),
        micronutrients=[
            Micronutrient(name="Protein", amount=protein, unit="g"),
            Micronutrient(name="Calcium", amount=100, unit="mg")
        ]
    )

class TestMealSuggestion:
    """Test cases for the meal suggestion endpoint."""
    
    def test_suggest_meal_with_no_meals_consumed(self):
        """Test meal suggestion when no meals have been consumed today."""
        request_data = {
            "todays_meals": [],
            "daily_protein_goal": 150.0
        }
        
        with patch('app.suggest_meal') as mock_suggest:
            mock_suggest.return_value = [
                {
                    "mealName": "Grilled chicken breast with quinoa and steamed broccoli",
                    "servingSize": {"qty": 1, "unit": "plate", "grams": 400},
                    "ingredients": "Grilled chicken breast (6oz), quinoa (1 cup cooked), steamed broccoli (1 cup)",
                    "category": "Poultry",
                    "macros": {
                        "calories": 500,
                        "protein": 45,
                        "carbohydrates": {"total": 40, "net": 35, "fiber": 5, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                        "fat": {"total": 10, "saturated": 2, "monounsaturated": 4, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 95}
                    },
                    "micronutrients": []
                }
            ]
            
            response = client.post("/suggest-meal", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            meal = data[0]
            assert meal["mealName"] == "Grilled chicken breast with quinoa and steamed broccoli"
            assert meal["macros"]["protein"] == 45
            assert meal["macros"]["calories"] == 500
            assert meal["micronutrients"] == []
    
    def test_suggest_meal_with_partial_protein_intake(self):
        """Test meal suggestion when some protein has been consumed."""
        meals = [
            create_sample_meal("Oatmeal with berries", 8.0, 300),
            create_sample_meal("Greek yogurt", 15.0, 150)
        ]
        
        request_data = {
            "todays_meals": [meal.model_dump() for meal in meals],
            "daily_protein_goal": 120.0
        }
        
        with patch('app.suggest_meal') as mock_suggest:
            mock_suggest.return_value = [
                {
                    "mealName": "Grilled salmon with sweet potato and asparagus",
                    "servingSize": {"qty": 1, "unit": "plate", "grams": 350},
                    "ingredients": "Grilled salmon (5oz), sweet potato (1 medium), asparagus (1 cup)",
                    "category": "Seafood",
                    "macros": {
                        "calories": 450,
                        "protein": 35,
                        "carbohydrates": {"total": 35, "net": 30, "fiber": 5, "sugar": 8, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                        "fat": {"total": 15, "saturated": 3, "monounsaturated": 6, "polyunsaturated": 4, "omega3": 2, "omega6": 2, "cholesterol": 60}
                    },
                    "micronutrients": []
                }
            ]
            
            response = client.post("/suggest-meal", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            meal = data[0]
            assert meal["macros"]["protein"] == 35  # Remaining protein: 120 - 23 = 97
    
    def test_suggest_meal_protein_goal_already_met(self):
        """Test meal suggestion when protein goal is already met or exceeded."""
        meals = [
            create_sample_meal("Protein shake", 30.0, 200),
            create_sample_meal("Chicken breast", 40.0, 300),
            create_sample_meal("Greek yogurt", 20.0, 150)
        ]
        
        request_data = {
            "todays_meals": [meal.model_dump() for meal in meals],
            "daily_protein_goal": 80.0
        }
        
        with patch('app.suggest_meal') as mock_suggest:
            mock_suggest.return_value = [
                {
                    "mealName": "Light salad with mixed greens and vegetables",
                    "servingSize": {"qty": 1, "unit": "bowl", "grams": 200},
                    "ingredients": "Mixed greens (2 cups), vegetables (1 cup), olive oil dressing",
                    "category": "Vegetable",
                    "macros": {
                        "calories": 100,
                        "protein": 5,
                        "carbohydrates": {"total": 15, "net": 10, "fiber": 5, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                        "fat": {"total": 5, "saturated": 1, "monounsaturated": 3, "polyunsaturated": 1, "omega3": 0.2, "omega6": 0.8, "cholesterol": 0}
                    },
                    "micronutrients": []
                }
            ]
            
            response = client.post("/suggest-meal", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            meal = data[0]
            assert meal["macros"]["protein"] == 5  # Light meal since goal exceeded
    
    def test_suggest_meal_invalid_request(self):
        """Test meal suggestion with invalid request data."""
        # Missing daily_protein_goal
        request_data = {
            "todays_meals": []
        }
        
        response = client.post("/suggest-meal", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_suggest_meal_negative_protein_goal(self):
        """Test meal suggestion with negative protein goal."""
        request_data = {
            "todays_meals": [],
            "daily_protein_goal": -10.0
        }
        
        with patch('app.suggest_meal') as mock_suggest:
            mock_suggest.return_value = [
                {
                    "mealName": "Light snack - mixed nuts",
                    "servingSize": {"qty": 1, "unit": "handful", "grams": 30},
                    "ingredients": "Mixed nuts (almonds, cashews, walnuts)",
                    "category": "Nut",
                    "macros": {
                        "calories": 150,
                        "protein": 5,
                        "carbohydrates": {"total": 8, "net": 6, "fiber": 2, "sugar": 2, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                        "fat": {"total": 12, "saturated": 1.5, "monounsaturated": 7, "polyunsaturated": 3, "omega3": 1, "omega6": 2, "cholesterol": 0}
                    },
                    "micronutrients": []
                }
            ]
            
            response = client.post("/suggest-meal", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

class TestImageAnalysis:
    """Test cases for image analysis endpoints."""
    
    @patch('app.analyze_image')
    def test_analyze_image_url(self, mock_analyze):
        """Test image analysis with URL."""
        mock_analyze.return_value = [
            {
                "mealName": "Grilled Chicken Salad",
                "servingSize": {"qty": 1, "unit": "plate", "grams": 350},
                "ingredients": "Grilled chicken, lettuce, tomatoes, olive oil",
                "macros": {
                    "calories": 350,
                    "protein": 30,
                    "carbohydrates": {"total": 8, "net": 6, "fiber": 2, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 20, "saturated": 4, "monounsaturated": 12, "polyunsaturated": 3, "omega3": 1, "omega6": 2, "cholesterol": 50}
                },
                "micronutrients": [
                    {"name": "Protein", "amount": 30, "unit": "g"},
                    {"name": "Vitamin A", "amount": 900, "unit": "µg"}
                ]
            }
        ]
        
        request_data = {"image_url": "https://example.com/food.jpg"}
        response = client.post("/analyze-image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["mealName"] == "Grilled Chicken Salad"
    
    @patch('app.analyze_image')
    def test_analyze_image_upload(self, mock_analyze):
        """Test image analysis with file upload."""
        mock_analyze.return_value = [
            {
                "mealName": "Pasta with Marinara",
                "servingSize": {"qty": 1, "unit": "bowl", "grams": 300},
                "ingredients": "Pasta, tomato sauce, herbs",
                "macros": {
                    "calories": 400,
                    "protein": 15,
                    "carbohydrates": {"total": 60, "net": 55, "fiber": 5, "sugar": 8, "addedSugar": 2, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 12, "saturated": 2, "monounsaturated": 6, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 0}
                },
                "micronutrients": [
                    {"name": "Protein", "amount": 15, "unit": "g"},
                    {"name": "Fiber", "amount": 5, "unit": "g"}
                ]
            }
        ]
        
        # Create a dummy image file
        files = {"file": ("test.jpg", b"fake_image_data", "image/jpeg")}
        response = client.post("/analyze-image-upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["mealName"] == "Pasta with Marinara"

class TestTranscriptionAnalysis:
    """Test cases for transcription analysis endpoint."""
    
    @patch('app.analyze_transcription')
    def test_analyze_transcription_single_meal(self, mock_analyze):
        """Test transcription analysis with single meal."""
        mock_analyze.return_value = [
            {
                "mealName": "Breakfast Oatmeal",
                "servingSize": {"qty": 1, "unit": "bowl", "grams": 250},
                "ingredients": "Oats, milk, honey, berries",
                "category": "Grain",
                "macros": {
                    "calories": 300,
                    "protein": 12,
                    "carbohydrates": {"total": 50, "net": 45, "fiber": 5, "sugar": 15, "addedSugar": 10, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 8, "saturated": 2, "monounsaturated": 3, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 5}
                },
                "micronutrients": [
                    {"name": "Protein", "amount": 12, "unit": "g"},
                    {"name": "Fiber", "amount": 5, "unit": "g"}
                ]
            }
        ]
        
        request_data = {"transcription": "I had oatmeal with milk and berries for breakfast"}
        response = client.post("/analyze-transcription", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["mealName"] == "Breakfast Oatmeal"
        assert data[0]["category"] == "Grain"
    
    @patch('app.analyze_transcription')
    def test_analyze_transcription_multiple_meals(self, mock_analyze):
        """Test transcription analysis with multiple meals."""
        mock_analyze.return_value = [
            {
                "mealName": "Grilled Chicken",
                "servingSize": {"qty": 1, "unit": "piece", "grams": 200},
                "ingredients": "Chicken breast, olive oil, herbs",
                "category": "Poultry",
                "macros": {
                    "calories": 250,
                    "protein": 35,
                    "carbohydrates": {"total": 2, "net": 2, "fiber": 0, "sugar": 0, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 12, "saturated": 3, "monounsaturated": 6, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 80}
                },
                "micronutrients": [
                    {"name": "Protein", "amount": 35, "unit": "g"},
                    {"name": "Iron", "amount": 2, "unit": "mg"}
                ]
            },
            {
                "mealName": "Rice and Vegetables",
                "servingSize": {"qty": 1, "unit": "plate", "grams": 300},
                "ingredients": "White rice, mixed vegetables, soy sauce",
                "category": "Grain",
                "macros": {
                    "calories": 200,
                    "protein": 5,
                    "carbohydrates": {"total": 40, "net": 38, "fiber": 2, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 3, "saturated": 0.5, "monounsaturated": 1, "polyunsaturated": 1, "omega3": 0.2, "omega6": 0.8, "cholesterol": 0}
                },
                "micronutrients": [
                    {"name": "Protein", "amount": 5, "unit": "g"},
                    {"name": "Fiber", "amount": 2, "unit": "g"}
                ]
            }
        ]
        
        request_data = {"transcription": "I had grilled chicken for lunch, then rice with vegetables for dinner"}
        response = client.post("/analyze-transcription", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["mealName"] == "Grilled Chicken"
        assert data[1]["mealName"] == "Rice and Vegetables"

class TestOutputValidation:
    """Test cases for output structure validation."""
    
    @patch('app.analyze_image')
    def test_image_analysis_output_structure(self, mock_analyze):
        """Test that image analysis returns correct structure with empty micronutrients."""
        mock_analyze.return_value = [
            {
                "mealName": "Grilled Chicken Salad",
                "servingSize": {"qty": 1, "unit": "plate", "grams": 350},
                "ingredients": "Grilled chicken, lettuce, tomatoes, olive oil",
                "macros": {
                    "calories": 350,
                    "protein": 30,
                    "carbohydrates": {"total": 8, "net": 6, "fiber": 2, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 20, "saturated": 4, "monounsaturated": 12, "polyunsaturated": 3, "omega3": 1, "omega6": 2, "cholesterol": 50}
                },
                "micronutrients": []
            }
        ]
        
        request_data = {"image_url": "https://example.com/food.jpg"}
        response = client.post("/analyze-image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        
        meal = data[0]
        # Check required fields exist
        assert "mealName" in meal
        assert "servingSize" in meal
        assert "ingredients" in meal
        assert "macros" in meal
        assert "micronutrients" in meal
        
        # Check micronutrients is empty array
        assert meal["micronutrients"] == []
        
        # Check macros structure
        macros = meal["macros"]
        assert "calories" in macros
        assert "protein" in macros
        assert "carbohydrates" in macros
        assert "fat" in macros
        
        # Check serving size structure
        serving_size = meal["servingSize"]
        assert "qty" in serving_size
        assert "unit" in serving_size
        assert "grams" in serving_size
    
    @patch('app.analyze_transcription')
    def test_transcription_analysis_output_structure(self, mock_analyze):
        """Test that transcription analysis returns correct structure with empty micronutrients."""
        mock_analyze.return_value = [
            {
                "mealName": "Breakfast Oatmeal",
                "servingSize": {"qty": 1, "unit": "bowl", "grams": 250},
                "ingredients": "Oats, milk, honey, berries",
                "category": "Grain",
                "macros": {
                    "calories": 300,
                    "protein": 12,
                    "carbohydrates": {"total": 50, "net": 45, "fiber": 5, "sugar": 15, "addedSugar": 10, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 8, "saturated": 2, "monounsaturated": 3, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 5}
                },
                "micronutrients": []
            }
        ]
        
        request_data = {"transcription": "I had oatmeal with milk and berries for breakfast"}
        response = client.post("/analyze-transcription", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        
        meal = data[0]
        # Check required fields exist
        assert "mealName" in meal
        assert "servingSize" in meal
        assert "ingredients" in meal
        assert "category" in meal
        assert "macros" in meal
        assert "micronutrients" in meal
        
        # Check micronutrients is empty array
        assert meal["micronutrients"] == []
        
        # Check macros structure
        macros = meal["macros"]
        assert "calories" in macros
        assert "protein" in macros
        assert "carbohydrates" in macros
        assert "fat" in macros
        
        # Check carbohydrates structure
        carbs = macros["carbohydrates"]
        required_carb_fields = ["total", "net", "fiber", "sugar", "addedSugar", "sugarAlcohols", "allulose"]
        for field in required_carb_fields:
            assert field in carbs
        
        # Check fat structure
        fat = macros["fat"]
        required_fat_fields = ["total", "saturated", "monounsaturated", "polyunsaturated", "omega3", "omega6", "cholesterol"]
        for field in required_fat_fields:
            assert field in fat
    
    @patch('app.suggest_meal')
    def test_meal_suggestion_output_structure(self, mock_suggest):
        """Test that meal suggestion returns correct structure (same as transcription)."""
        mock_suggest.return_value = [
            {
                "mealName": "Grilled chicken breast with quinoa and steamed broccoli",
                "servingSize": {"qty": 1, "unit": "plate", "grams": 400},
                "ingredients": "Grilled chicken breast (6oz), quinoa (1 cup cooked), steamed broccoli (1 cup)",
                "category": "Poultry",
                "macros": {
                    "calories": 450,
                    "protein": 35,
                    "carbohydrates": {"total": 40, "net": 35, "fiber": 5, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 10, "saturated": 2, "monounsaturated": 4, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 95}
                },
                "micronutrients": []
            }
        ]
        
        request_data = {
            "todays_meals": [],
            "daily_protein_goal": 120.0
        }
        
        response = client.post("/suggest-meal", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check it's a list with one meal
        assert isinstance(data, list)
        assert len(data) == 1
        
        meal = data[0]
        # Check required fields exist (same as transcription)
        required_fields = ["mealName", "servingSize", "ingredients", "category", "macros", "micronutrients"]
        for field in required_fields:
            assert field in meal
        
        # Check micronutrients is empty
        assert meal["micronutrients"] == []
        
        # Check data types
        assert isinstance(meal["mealName"], str)
        assert isinstance(meal["macros"]["protein"], (int, float))
        assert isinstance(meal["macros"]["calories"], (int, float))
    
    @patch('app.analyze_image')
    def test_multiple_meals_output_structure(self, mock_analyze):
        """Test that multiple meals are returned with correct structure."""
        mock_analyze.return_value = [
            {
                "mealName": "Grilled Chicken",
                "servingSize": {"qty": 1, "unit": "piece", "grams": 200},
                "ingredients": "Chicken breast, olive oil, herbs",
                "macros": {
                    "calories": 250,
                    "protein": 35,
                    "carbohydrates": {"total": 2, "net": 2, "fiber": 0, "sugar": 0, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 12, "saturated": 3, "monounsaturated": 6, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 80}
                },
                "micronutrients": []
            },
            {
                "mealName": "Rice and Vegetables",
                "servingSize": {"qty": 1, "unit": "plate", "grams": 300},
                "ingredients": "White rice, mixed vegetables, soy sauce",
                "macros": {
                    "calories": 200,
                    "protein": 5,
                    "carbohydrates": {"total": 40, "net": 38, "fiber": 2, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                    "fat": {"total": 3, "saturated": 0.5, "monounsaturated": 1, "polyunsaturated": 1, "omega3": 0.2, "omega6": 0.8, "cholesterol": 0}
                },
                "micronutrients": []
            }
        ]
        
        request_data = {"image_url": "https://example.com/food.jpg"}
        response = client.post("/analyze-image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        
        # Check both meals have correct structure
        for meal in data:
            assert "mealName" in meal
            assert "servingSize" in meal
            assert "ingredients" in meal
            assert "macros" in meal
            assert "micronutrients" in meal
            assert meal["micronutrients"] == []
    
    def test_pydantic_model_validation(self):
        """Test that response data can be validated against Pydantic models."""
        from app import MealIMAGE, MealTRANSCRIPTION, ServingSize, Macronutrients, Carbohydrates, Fat, Micronutrient
        
        # Test MealIMAGE model
        meal_image_data = {
            "mealName": "Test Meal",
            "servingSize": {"qty": 1, "unit": "plate", "grams": 300},
            "ingredients": "Test ingredients",
            "macros": {
                "calories": 400,
                "protein": 25,
                "carbohydrates": {"total": 30, "net": 25, "fiber": 5, "sugar": 8, "addedSugar": 2, "sugarAlcohols": 0, "allulose": 0},
                "fat": {"total": 15, "saturated": 3, "monounsaturated": 6, "polyunsaturated": 3, "omega3": 1, "omega6": 2, "cholesterol": 20}
            },
            "micronutrients": []
        }
        
        # Should not raise validation error
        meal_image = MealIMAGE(**meal_image_data)
        assert meal_image.mealName == "Test Meal"
        assert meal_image.micronutrients == []
        
        # Test MealTRANSCRIPTION model
        meal_transcription_data = {
            "mealName": "Test Meal",
            "servingSize": {"qty": 1, "unit": "plate", "grams": 300},
            "ingredients": "Test ingredients",
            "category": "Test",
            "macros": {
                "calories": 400,
                "protein": 25,
                "carbohydrates": {"total": 30, "net": 25, "fiber": 5, "sugar": 8, "addedSugar": 2, "sugarAlcohols": 0, "allulose": 0},
                "fat": {"total": 15, "saturated": 3, "monounsaturated": 6, "polyunsaturated": 3, "omega3": 1, "omega6": 2, "cholesterol": 20}
            },
            "micronutrients": []
        }
        
        # Should not raise validation error
        meal_transcription = MealTRANSCRIPTION(**meal_transcription_data)
        assert meal_transcription.mealName == "Test Meal"
        assert meal_transcription.category == "Test"
        assert meal_transcription.micronutrients == []
        
        # Test that meal suggestion uses same model as transcription
        meal_suggestion_data = {
            "mealName": "Suggested Meal",
            "servingSize": {"qty": 1, "unit": "plate", "grams": 400},
            "ingredients": "Test suggestion ingredients",
            "category": "Poultry",
            "macros": {
                "calories": 450,
                "protein": 35,
                "carbohydrates": {"total": 40, "net": 35, "fiber": 5, "sugar": 3, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0},
                "fat": {"total": 10, "saturated": 2, "monounsaturated": 4, "polyunsaturated": 2, "omega3": 0.5, "omega6": 1.5, "cholesterol": 95}
            },
            "micronutrients": []
        }
        
        # Should not raise validation error (uses MealTRANSCRIPTION model)
        meal_suggestion = MealTRANSCRIPTION(**meal_suggestion_data)
        assert meal_suggestion.mealName == "Suggested Meal"
        assert meal_suggestion.macros.protein == 35

class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_meal_suggestion_api_error(self):
        """Test meal suggestion when API call fails."""
        request_data = {
            "todays_meals": [],
            "daily_protein_goal": 100.0
        }
        
        with patch('app.suggest_meal', side_effect=Exception("API Error")):
            response = client.post("/suggest-meal", json=request_data)
            assert response.status_code == 500
    
    def test_analyze_image_api_error(self):
        """Test image analysis when API call fails."""
        request_data = {"image_url": "https://example.com/food.jpg"}
        
        with patch('app.analyze_image', side_effect=Exception("API Error")):
            response = client.post("/analyze-image", json=request_data)
            assert response.status_code == 500
    
    def test_analyze_transcription_api_error(self):
        """Test transcription analysis when API call fails."""
        request_data = {"transcription": "I had breakfast"}
        
        with patch('app.analyze_transcription', side_effect=Exception("API Error")):
            response = client.post("/analyze-transcription", json=request_data)
            assert response.status_code == 500

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
