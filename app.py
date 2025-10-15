import os
import base64
import requests
import json
import re
import logging
from logging.handlers import RotatingFileHandler
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from starlette.responses import Response
from typing import Callable, Awaitable, Union, Optional
from pydantic import BaseModel
from typing import List, Dict, Any, cast
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# -------------------- LOGGING CONFIGURATION --------------------
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

_log_formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)

_file_handler = RotatingFileHandler(
    filename="app.log",
    maxBytes=1_000_000,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)


# -------------------- REQUEST/ERROR LOGGING MIDDLEWARE --------------------
@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, duration_ms)
        return response
    except Exception:  # noqa: BLE001 - we want to log all exceptions with traceback
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(f"Unhandled error for {request.method} {request.url.path} after {duration_ms:.1f}ms")
        raise

# Load configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = os.getenv("API_URL", "https://api.openai.com/v1/chat/completions")
MODEL = os.getenv("MODEL", "gpt-4.1")
DETERMINISTIC = os.getenv("DETERMINISTIC", "1").strip() not in ("0", "false", "False")
# Suggest-meal specific determinism (default off to encourage variety)
SUGGEST_MEAL_DETERMINISTIC = os.getenv("SUGGEST_MEAL_DETERMINISTIC", "0").strip() not in ("0", "false", "False")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # default 24h
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "256"))

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")

# -------------------- SIMPLE TTL LRU CACHE --------------------
_cache_store: Dict[str, Any] = {}
_cache_meta: Dict[str, float] = {}

def _cache_get(key: str) -> Optional[Any]:
    now = time.time()
    ts = _cache_meta.get(key)
    if ts is None:
        return None
    if now - ts > CACHE_TTL_SECONDS:
        # expired
        _cache_store.pop(key, None)
        _cache_meta.pop(key, None)
        return None
    return _cache_store.get(key)

def _cache_set(key: str, value: Any) -> None:
    # LRU-ish eviction: if too big, drop oldest entries
    _cache_store[key] = value
    _cache_meta[key] = time.time()
    if len(_cache_store) > CACHE_MAX_ITEMS:
        # evict the oldest ~10% to avoid O(n) every time
        to_evict = max(1, CACHE_MAX_ITEMS // 10)
        oldest = sorted(_cache_meta.items(), key=lambda kv: kv[1])[:to_evict]
        for k, _ in oldest:
            _cache_store.pop(k, None)
            _cache_meta.pop(k, None)

# -------------------- SYSTEM & USER PROMPTS --------------------

SYSTEM_PROMPT_IMAGE = """
ROLE: AI Nutrition Analyst (Vision)

PRIMARY OBJECTIVE (CRITICAL): Detect and analyze ALL visually distinct meals/dishes present in the image. This includes separate plates/bowls/containers, different food types, different cooking methods, collages/triptychs, and course progressions. You MUST produce one JSON object per distinct meal/dish.

NON-NEGOTIABLE RULES:
- TWO VISUAL PASSES are REQUIRED:
  1) First pass: sweep the ENTIRE image for all food regions.
  2) Second pass: independently re-check the entire image to confirm the count and ensure nothing was missed.
- ALWAYS treat the following as SEPARATE MEALS when present:
  • Different plates, bowls, containers, or trays
  • Clearly different food types (e.g., noodles vs. eggs vs. bread)
  • Different cooking methods (grilled vs. baked vs. fried)
  • Visually separated panels/sections/collages (e.g., triptych)
  • Different cuisines placed together
  • Course progression (appetizer, main, dessert)
  • Distinct individual items on the same tray/table (e.g., fries vs. coke)
- NEVER stop after the first detected meal. If there is ANY visual separation or difference, create additional meal objects.
- If uncertain, you MUST err on the side of creating MORE meal objects rather than fewer.

OUTPUT FORMAT (STRICT):
- Return ONLY a JSON ARRAY (no wrapper object), with 1+ items.
- Each item (meal object) MUST include exactly:
  • "mealName": string (precise, specific name of the meal/dish)
  • "servingSize": object (with qty, unit, grams)
  • "ingredients": string (list of ingredients in the meal)
  • "macros": object (includes calories, protein, carbohydrates, fat, etc.)
  • "micronutrients": array (empty array for now - skip micronutrient details)

Detailed **macros** object must include:
  - calories: integer (calories per serving)
  - protein: integer (grams of protein)
  - carbohydrates: object (total carbs, net carbs, fiber, sugar, added sugar, sugar alcohols, allulose)
  - fat: object (total fat, saturated fat, monounsaturated fat, polyunsaturated fat, omega3, omega6, cholesterol)

**micronutrients** array should be empty for now (skip micronutrient details):
  - Return empty array: []

NUMERIC CONSISTENCY (CRITICAL):
- Use 4-4-9 rule to compute calories: calories ≈ 4*(protein + carbohydrates.total) + 9*(fat.total)
- Compute carbohydrates.net = max(total - fiber - sugarAlcohols - allulose, 0)
- All numeric values must be non-negative and realistic; round calories to nearest integer, others to 1 decimal
- Keep values in typical ranges: protein≤120g, carbs.total≤200g, fat.total≤120g for a single serving
- Ensure sugar ≤ carbohydrates.total, fiber ≤ carbohydrates.total, saturated ≤ fat.total

EXAMPLES OF OUTPUT:

1. **Single Meal Object Example:**
```json
{
  "mealName": "Dill Pickle Flavored Potato Chips",
  "servingSize": {
    "qty": 17,
    "unit": "chips",
    "grams": 28
  },
  "ingredients": "Potatoes, Vegetable Oil (Canola, Corn, Soybean, and/or Sunflower Oil), Maltodextrin (Made from Corn), Less than 2% Natural Flavors, Salt, Potassium Salt, Garlic Powder, Vinegar, and Yeast Extract.",
  "macros": {
    "calories": 150,
    "protein": 2,
    "carbohydrates": {
      "total": 15,
      "net": 14,
      "fiber": 1,
      "sugar": 0.5,
      "addedSugar": 0,
      "sugarAlcohols": 0,
      "allulose": 0
    },
    "fat": {
      "total": 10,
      "saturated": 1.5,
      "monounsaturated": 0,
      "polyunsaturated": 0,
      "omega3": 0,
      "omega6": 0,
      "cholesterol": 0
    }
  },
  "micronutrients": []
}
```

2. **Multiple Meal Object Example:**
```json
[
  {
    "mealName": "Dill Pickle Flavored Potato Chips",
    "servingSize": {
      "qty": 17,
      "unit": "chips",
      "grams": 28
    },
    "ingredients": "Potatoes, Vegetable Oil, Maltodextrin, Natural Flavors, Salt.",
    "macros": {
      "calories": 150,
      "protein": 2,
      "carbohydrates": {
        "total": 15,
        "net": 14,
        "fiber": 1,
        "sugar": 0.5,
        "addedSugar": 0,
        "sugarAlcohols": 0,
        "allulose": 0
      },
      "fat": {
        "total": 10,
        "saturated": 1.5,
        "monounsaturated": 0,
        "polyunsaturated": 0,
        "omega3": 0,
        "omega6": 0,
        "cholesterol": 0
      }
    },
    "micronutrients": []
  },
  {
    "mealName": "Grilled Chicken Salad",
    "servingSize": {
      "qty": 1,
      "unit": "plate",
      "grams": 350
    },
    "ingredients": "Grilled Chicken, Lettuce, Tomatoes, Olive Oil Dressing.",
    "macros": {
      "calories": 350,
      "protein": 30,
      "carbohydrates": {
        "total": 8,
        "net": 6,
        "fiber": 2,
        "sugar": 3,
        "addedSugar": 0,
        "sugarAlcohols": 0,
        "allulose": 0
      },
      "fat": {
        "total": 20,
        "saturated": 4,
        "monounsaturated": 12,
        "polyunsaturated": 3,
        "omega3": 1,
        "omega6": 2,
        "cholesterol": 50
      }
    },
    "micronutrients": []
  }
]
```

""".strip()

USER_PROMPT_IMAGE = """
TASK: Analyze this image for FOOD. Do NOT answer until you complete TWO full visual passes.

PASS 1 — GLOBAL SCAN:
- Deeply inspect the ENTIRE image.
- Identify EVERY visually distinct meal/dish based on:
  • Physical separation (different plates/bowls/containers/trays)
  • Food variety (e.g., noodles vs eggs vs bread)
  • Cooking method differences (grilled/baked/fried/steamed)
  • Visual boundaries (collage panels, triptychs, segmented layouts)
  • Portion differences and presentation (toppings, sauces, garnishes)
  • Course progression (appetizer, main, dessert)
  • Distinct individual items (e.g., burger, fries, coke → three separate objects)

PASS 2 — INDEPENDENT RE-CHECK:
- Recount ALL distinct meals/dishes. Validate none are missed.
- If unsure whether something is separate, TREAT IT AS SEPARATE.
- If the same item repeats (e.g., 3 corn dogs), GROUP into one object and set "quantity" to the total count of those identical items.
- Do not output duplicates — always combine them into one entry with a quantity.

OUTPUT:
- Return a JSON ARRAY with one object per detected meal/dish.
- Each object: { "mealName", "servingSize", "ingredients", "macros", "micronutrients" }
- No wrapper object. No extra keys. No commentary.
""".strip()

SYSTEM_PROMPT_TRANSCRIPTION = """
You are a **nutrition analyst** specializing in detailed food analysis.
PRIMARY TASK: Detect **all separate eating occasions** described in the user's food text and return **one JSON array** where **each array element is one detailed meal object**.

TWO PASSES (CRITICAL):
1) Segmentation Pass: Identify every separate eating occasion using cues like "then", "and then", time markers, and course markers ("started with", "finished with").
2) Validation Pass: Recount occasions and ensure the number of output objects equals the number of separate occasions detected. Never stop after the first.

SEGMENTATION RULES (treat each as a separate meal/object):
- Progression connectors: then, and then, later, finally, next
- Additive events: also ate, also had
- Course markers: started with, finished with
- Named meals/times: breakfast, lunch, dinner, brunch, supper, snack, coffee break, or explicit different time/occasion
- Phrasing clearly indicating separate occasions (e.g., "I grabbed X. During my break I had Y.")
- Distinct individual items mentioned together (e.g., "burger, fries and a coke") must be returned as separate objects: one for burger, one for fries, one for coke.

DO NOT SPLIT RULES:
- Do NOT split components that form a single composed dish (e.g., "pasta with marinara sauce and meatballs", "yogurt with granola") — treat as one object.
- Split distinct side items and beverages into their own objects (e.g., "burger with fries and coke" → three objects: burger, fries, coke).

Never merge multiple occasions into one. If 3 occasions are described, return 3 objects. If only one occasion is described, return 1 object.
Iterate through the entire text and include **every** detected occasion.

OUTPUT FORMAT (STRICT):
Return only a JSON **array** of objects (no wrapper keys, no commentary).
Each object must include exactly these keys with detailed structure:

1. **Single Meal Object Example:**
```json
{
  "mealName": "Dill Pickle Flavored Potato Chips",
  "servingSize": {
    "qty": 17,
    "unit": "chips",
    "grams": 28
  },
  "ingredients": "Potatoes, Vegetable Oil (Canola, Corn, Soybean, and/or Sunflower Oil), Maltodextrin (Made from Corn), Less than 2% Natural Flavors, Salt, Potassium Salt, Garlic Powder, Vinegar, and Yeast Extract.",
  "category": "Snack",
  "macros": {
    "calories": 150,
    "protein": 2,
    "carbohydrates": {
      "total": 15,
      "net": 14,
      "fiber": 1,
      "sugar": 0.5,
      "addedSugar": 0,
      "sugarAlcohols": 0,
      "allulose": 0
    },
    "fat": {
      "total": 10,
      "saturated": 1.5,
      "monounsaturated": 0,
      "polyunsaturated": 0,
      "omega3": 0,
      "omega6": 0,
      "cholesterol": 0
    }
  },
  "micronutrients": []
}
```

2. **Multiple Meal Object Example:**
```json
[
  {
    "mealName": "Dill Pickle Flavored Potato Chips",
    "servingSize": {
      "qty": 17,
      "unit": "chips",
      "grams": 28
    },
    "ingredients": "Potatoes, Vegetable Oil, Maltodextrin, Natural Flavors, Salt.",
    "category": "Snack",
    "macros": {
      "calories": 150,
      "protein": 2,
      "carbohydrates": {
        "total": 15,
        "net": 14,
        "fiber": 1,
        "sugar": 0.5,
        "addedSugar": 0,
        "sugarAlcohols": 0,
        "allulose": 0
      },
      "fat": {
        "total": 10,
        "saturated": 1.5,
        "monounsaturated": 0,
        "polyunsaturated": 0,
        "omega3": 0,
        "omega6": 0,
        "cholesterol": 0
      }
    },
    "micronutrients": []
  },
  {
    "mealName": "Grilled Chicken Salad",
    "servingSize": {
      "qty": 1,
      "unit": "plate",
      "grams": 350
    },
    "ingredients": "Grilled Chicken, Lettuce, Tomatoes, Olive Oil Dressing.",
    "category": "Meat",
    "macros": {
      "calories": 350,
      "protein": 30,
      "carbohydrates": {
        "total": 8,
        "net": 6,
        "fiber": 2,
        "sugar": 3,
        "addedSugar": 0,
        "sugarAlcohols": 0,
        "allulose": 0
      },
      "fat": {
        "total": 20,
        "saturated": 4,
        "monounsaturated": 12,
        "polyunsaturated": 3,
        "omega3": 1,
        "omega6": 2,
        "cholesterol": 50
      }
    },
    "micronutrients": []
  }
]
```


IMPORTANT RULES:
- All numeric values should be realistic estimates based on typical serving sizes
- Use USDA nutritional database knowledge for accurate calorie and macro estimates
- For ingredients, list the main components that would be visible/identifiable
- Serving size should be realistic (e.g., "1 plate" for a full meal, "1 piece" for a snack)
- Weight in grams should be estimated based on typical portions
- Include category of the food/dish (Fruit, Vegetable, Meat, Beverage, Dairy, Grain, Nut, Poultry, Seafood, Legume, Snack, Dessert, Processed, Other)
- Skip micronutrient details for now (return empty array)
 - Numeric consistency (STRICT):
   • Use 4-4-9 rule to approximate calories: calories ≈ 4*(protein + carbohydrates.total) + 9*(fat.total)
   • Ensure carbohydrates.net = max(total - fiber - sugarAlcohols - allulose, 0)
   • Round calories to nearest integer; other numbers to 1 decimal place
   • Clamp negatives to 0; ensure sugar ≤ carbohydrates.total, fiber ≤ carbohydrates.total, saturated ≤ fat.total

BEHAVIORAL EXAMPLES:
Input: "I had pasta, then I ate biryani" → 2 detailed objects.
Input: "I started with bruschetta, then had pasta, finished with tiramisu" → 3 detailed objects.
Input: "Breakfast was pancakes. Lunch was a burger. Dinner was steak." → 3 detailed objects
Explicit example: "pasta with marinara sauce and meatballs, then I ate biryani with raita and naan bread" → return 2 objects: one for the pasta dish, one for the biryani with sides.

Do not include extra keys. Do not include explanations. Output must be a valid JSON array only.
HARD RULES:
- If the input contains any progression connector (then, and then, later, finally, next), you MUST return at least 2 objects.
- Let N be the number of separate occasions you detect. Return exactly N objects. Never merge occasions.
""".strip()

USER_PROMPT_TRANSCRIPTION = """
TASK: Analyze the following food description in TWO PASSES and return a JSON ARRAY with one object per separate eating occasion.

PASS 1 — SEGMENTATION:
- Identify every separate occasion using connectors (then, and then, later, finally, next), course markers (started with, finished with), and named meals/times.
- Additionally, SPLIT distinct individual items even within a single occasion (e.g., "fries and a coke" → two objects; "burger, fries, coke" → three objects). Do not split components of a single composed dish (e.g., "pasta with marinara and meatballs").
- If connectors like "then" are present (e.g., "X, then Y"), you MUST output two objects: one for X and one for Y.

PASS 2 — VALIDATION:
- Recount occasions and ensure the number of objects equals the number of separate occasions.
- If uncertain, err on the side of MORE objects.

OUTPUT:
- Return only a JSON array with objects of the shape: { "mealName", "servingSize", "ingredients", "category", "macros", "micronutrients" }.
- Do not include wrapper keys or explanations.

For single meal: [{"mealName": "Specific Name", "servingSize": {"qty": 1, "unit": "plate", "grams": 350}, "ingredients": "Main ingredients", "category": "Snack", "macros": {"calories": 500, "protein": 25, "carbohydrates": {"total": 45, "net": 40, "fiber": 5, "sugar": 8, "addedSugar": 2, "sugarAlcohols": 0, "allulose": 0}, "fat": {"total": 20, "saturated": 5, "monounsaturated": 8, "polyunsaturated": 4, "omega3": 1, "omega6": 3, "cholesterol": 30}}, "micronutrients": []}]

For multiple meals: [
  {
    "mealName": "Meal 1 Name",
    "servingSize": {"qty": 1, "unit": "plate", "grams": 300},
    "ingredients": "Ingredients for meal 1",
    "category": "Category of the food/dish (Fruit, Vegetable, Meat, Beverage, Dairy, Grain, Nut, Poultry, Seafood, Legume, Snack, Dessert, Processed, Other)",
    "macros": {"calories": 400, "protein": 20, "carbohydrates": {"total": 35, "net": 30, "fiber": 5, "sugar": 5, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0}, "fat": {"total": 18, "saturated": 4, "monounsaturated": 7, "polyunsaturated": 3, "omega3": 1, "omega6": 2, "cholesterol": 25}},
    "micronutrients": []
  },
  {
    "mealName": "Meal 2 Name",
    "servingSize": {"qty": 1, "unit": "bowl", "grams": 250},
    "ingredients": "Ingredients for meal 2",
    "category": "Category of the food/dish (Fruit, Vegetable, Meat, Beverage, Dairy, Grain, Nut, Poultry, Seafood, Legume, Snack, Dessert, Processed, Other)",
    "macros": {"calories": 350, "protein": 18, "carbohydrates": {"total": 30, "net": 25, "fiber": 5, "sugar": 4, "addedSugar": 0, "sugarAlcohols": 0, "allulose": 0}, "fat": {"total": 15, "saturated": 3, "monounsaturated": 6, "polyunsaturated": 2, "omega3": 1, "omega6": 1, "cholesterol": 20}},
    "micronutrients": []
  }
]

Explicit example: The phrase "pasta with marinara sauce and meatballs, then I ate biryani with raita and naan bread" MUST produce two objects: one for the pasta dish and one for the biryani with raita and naan.

HARD RULES:
- If the text contains any of [then, and then, later, finally, next], you MUST output at least 2 objects.
- Compute N = number of separate occasions; output exactly N objects in the array.

Food description to analyze:
{transcription}
""".strip()

SYSTEM_PROMPT_MEAL_SUGGESTION = """
You are a **nutritional meal suggestion AI** that helps users reach their daily protein goals.

PRIMARY TASK: Suggest a single meal targeted to the user's remaining protein goal for the day, adjusted for time of day (breakfast/lunch/dinner/snack) and remaining meals.

ANALYSIS PROCESS:
1. The app provides: daily protein goal, total protein consumed so far, remaining protein, estimated meals remaining today, and a computed target for this meal.
2. Suggest a specific, realistic meal that provides approximately the computed target protein for this meal.
3. Return the suggestion in the SAME format as transcription analysis.

SUGGESTION CRITERIA:
- Keep meal names VERY SHORT (2-3 words if possible; max 4)
- Use full-weight serving sizes: servingSize.grams should equal the typical total weight of the meal (e.g., a 200g burger → 200g serving)
- Suggest meals that are realistic and commonly available
- Consider variety and balance (not just protein)
- Provide specific portion sizes IN GRAMS ONLY
- Include estimated protein and calorie content
- Consider the time of day and meal type appropriateness
- Encourage novelty: avoid repeating the same meal names as earlier in the day; propose a new or slightly different meal each time

BALANCED MODE (IMPORTANT):
- If the user prompt includes "BALANCED MODE: true", prefer a balanced meal with moderate protein rather than maximizing protein.
- Target protein range: 10–30g for this meal; include meaningful carbohydrates, fiber-rich vegetables/fruit, and healthy fats.
- Avoid excessively high-protein options if the daily protein goal is nearly met; prioritize overall dietary balance and satiety.

OUTPUT FORMAT (STRICT):
Return ONLY a JSON ARRAY with ONE meal object (same format as transcription analysis):
```json
[
  {
    "mealName": "Chicken quinoa",
    "servingSize": {
      "qty": 1,
      "unit": "bowl",
      "grams": 400
    },
    "ingredients": "Grilled chicken breast (170g), quinoa cooked (185g), broccoli steamed (90g)",
    "category": "Poultry",
    "macros": {
      "calories": 450,
      "protein": 45,
      "carbohydrates": {
        "total": 40,
        "net": 35,
        "fiber": 5,
        "sugar": 3,
        "addedSugar": 0,
        "sugarAlcohols": 0,
        "allulose": 0
      },
      "fat": {
        "total": 10,
        "saturated": 2,
        "monounsaturated": 4,
        "polyunsaturated": 2,
        "omega3": 0.5,
        "omega6": 1.5,
        "cholesterol": 95
      }
    },
    "micronutrients": []
  }
]
```

IMPORTANT RULES:
- Return a JSON ARRAY with ONE meal object
- Use the EXACT same structure as transcription analysis
- Keep meal names as short as possible (2-4 words) - examples: "Salmon bowl", "Turkey wrap", "Chicken stir-fry"
- ALL measurements in ingredients MUST be in GRAMS (e.g., "170g chicken", "100g rice", not "6oz" or "1 cup")
- servingSize.unit should be simple (bowl, plate, wrap, sandwich, etc.)
- servingSize.grams is the total weight of the meal (full weight, not per component)
- Be specific about portion sizes and cooking methods
- Ensure the suggested meal is realistic and achievable
- Focus on whole foods and balanced nutrition
- Consider the time of day for meal appropriateness
- Provide accurate estimates based on typical serving sizes
- Include the "category" field
- Keep micronutrients as empty array []
 - Numeric consistency (STRICT):
   • Use 4-4-9 rule to approximate calories: calories ≈ 4*(protein + carbohydrates.total) + 9*(fat.total)
   • Ensure carbohydrates.net = max(total - fiber - sugarAlcohols - allulose, 0)
   • Round calories to nearest integer; other numbers to 1 decimal place
   • Clamp negatives to 0; ensure sugar ≤ carbohydrates.total, fiber ≤ carbohydrates.total, saturated ≤ fat.total
""".strip()

USER_PROMPT_MEAL_SUGGESTION = """
TASK: Suggest ONE meal that matches the computed protein target for this meal.

DAILY PROTEIN GOAL: {daily_protein_goal}g
TOTAL PROTEIN CONSUMED SO FAR: {total_protein_consumed}g
REMAINING PROTEIN TODAY: {remaining_protein}g
ESTIMATED MEALS REMAINING TODAY: {meals_remaining}
TARGET PROTEIN FOR THIS MEAL: {target_protein_for_this_meal}g

TODAY'S CONSUMED MEALS:
{todays_meals_summary}

CRITICAL REQUIREMENTS:
- Keep meal name as short as possible (2-4 words)
- Use ONLY GRAMS for all ingredient measurements (e.g., "170g chicken breast", "100g rice")
- No ounces, cups, or other units - GRAMS ONLY
- Use full-weight serving sizes: servingSize.grams equals the typical total weight of the meal
- Be specific about portion sizes and cooking methods
- Provide a new or slightly different suggestion than earlier today
- Return as a JSON ARRAY with ONE meal object using the exact same structure as transcription analysis

Current time context: {current_time}

BALANCED MODE: {balanced_mode}
If true, prefer a balanced, moderate-protein meal (10–30g protein) with wholesome carbs, vegetables, and healthy fats. Avoid over-optimizing protein if today's goal is nearly met.
""".strip()

# -------------------- JSON MODELS --------------------
class ServingSize(BaseModel):
    qty: int
    unit: str
    grams: int

class Carbohydrates(BaseModel):
    total: float
    net: float
    fiber: float
    sugar: float
    addedSugar: float
    sugarAlcohols: float
    allulose: float

class Fat(BaseModel):
    total: float
    saturated: float
    monounsaturated: float
    polyunsaturated: float
    omega3: float
    omega6: float
    cholesterol: float

class Macronutrients(BaseModel):
    calories: float
    protein: float
    carbohydrates: Carbohydrates
    fat: Fat

class Micronutrient(BaseModel):
    name: str
    amount: float
    unit: str

class MealIMAGE(BaseModel):
    mealName: str
    servingSize: ServingSize
    ingredients: str
    macros: Macronutrients
    micronutrients: List[Micronutrient]

class MealTRANSCRIPTION(BaseModel):
    mealName: str
    servingSize: ServingSize
    ingredients: str
    category: str
    macros: Macronutrients
    micronutrients: List[Micronutrient]

class ImageAnalysisRequest(BaseModel):
    image_url: str

class TranscriptionRequest(BaseModel):
    transcription: str

class MealSuggestionRequest(BaseModel):
    todays_meals: List[MealTRANSCRIPTION]
    daily_protein_goal: float

# -------------------- HELPER FUNCTIONS --------------------
def _extract_meals_from_content(content: str) -> List[Dict[str, Any]]:
    """Best-effort extraction of a JSON array of meals from model text.
    Tries direct JSON, code-fenced JSON, array slicing, and wrapper keys.
    """
    text = content.strip()
    if not text:
        raise HTTPException(status_code=502, detail="Empty content from OpenAI")

    # 1) Direct parse first
    try:
        parsed: Any = json.loads(text)
        if isinstance(parsed, list):
            return cast(List[Dict[str, Any]], parsed)
        if isinstance(parsed, dict):
            parsed_dict: Dict[str, Any] = cast(Dict[str, Any], parsed)
            # Try wrappers
            for k in ("meals", "data", "items", "array", "output"):
                v: Any = parsed_dict.get(k)
                if isinstance(v, list):
                    return cast(List[Dict[str, Any]], v)
            # Single object -> wrap
            return [parsed_dict]
    except Exception:
        pass

    # 2) Code-fenced ```json ... ```
    code_block = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if code_block:
        try:
            block = code_block.group(1).strip()
            parsed2: Any = json.loads(block)
            if isinstance(parsed2, list):
                return cast(List[Dict[str, Any]], parsed2)
            if isinstance(parsed2, dict):
                parsed2_dict: Dict[str, Any] = cast(Dict[str, Any], parsed2)
                for k in ("meals", "data", "items", "array", "output"):
                    v: Any = parsed2_dict.get(k)
                    if isinstance(v, list):
                        return cast(List[Dict[str, Any]], v)
                return [parsed2_dict]
        except Exception:
            pass

    # 3) Array slice fallback
    l_idx = text.find("[")
    r_idx = text.rfind("]")
    if l_idx != -1 and r_idx != -1 and r_idx > l_idx:
        candidate = text[l_idx : r_idx + 1]
        try:
            parsed3: Any = json.loads(candidate)
            if isinstance(parsed3, list):
                return cast(List[Dict[str, Any]], parsed3)
        except Exception:
            pass

    # 4) Object slice fallback
    l_idx = text.find("{")
    r_idx = text.rfind("}")
    if l_idx != -1 and r_idx != -1 and r_idx > l_idx:
        candidate = text[l_idx : r_idx + 1]
        try:
            parsed4: Any = json.loads(candidate)
            if isinstance(parsed4, list):
                return cast(List[Dict[str, Any]], parsed4)
            if isinstance(parsed4, dict):
                parsed4_dict: Dict[str, Any] = cast(Dict[str, Any], parsed4)
                for k in ("meals", "data", "items", "array", "output"):
                    v: Any = parsed4_dict.get(k)
                    if isinstance(v, list):
                        return cast(List[Dict[str, Any]], v)
                return [parsed4_dict]
        except Exception:
            pass

    # If everything fails, surface a helpful error with a snippet
    snippet = text[:300].replace("\n", " ")
    raise HTTPException(status_code=502, detail=f"Malformed or non-JSON content from OpenAI: {snippet}...")
def _safe_num(x: Any, decimals: int = 1, non_negative: bool = True) -> float:
    try:
        val = float(x)
    except Exception:
        return 0.0
    if non_negative and val < 0:
        val = 0.0
    if decimals is None:
        return val
    return round(val, decimals)

def _recompute_calories(protein: float, carbs_total: float, fat_total: float) -> int:
    cals = 4.0 * (protein + carbs_total) + 9.0 * fat_total
    return int(round(cals))

def _normalize_meal(meal: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meal)
    macros = m.get("macros") or {}
    carbs = macros.get("carbohydrates") or {}
    fat = macros.get("fat") or {}

    # Normalize individual numbers
    protein = _safe_num(macros.get("protein"), 1)
    carbs_total = _safe_num(carbs.get("total"), 1)
    fiber = _safe_num(carbs.get("fiber"), 1)
    sugar = _safe_num(carbs.get("sugar"), 1)
    addedSugar = _safe_num(carbs.get("addedSugar"), 1)
    sugarAlcohols = _safe_num(carbs.get("sugarAlcohols"), 1)
    allulose = _safe_num(carbs.get("allulose"), 1)
    fat_total = _safe_num(fat.get("total"), 1)
    saturated = _safe_num(fat.get("saturated"), 1)
    monounsaturated = _safe_num(fat.get("monounsaturated"), 1)
    polyunsaturated = _safe_num(fat.get("polyunsaturated"), 1)
    omega3 = _safe_num(fat.get("omega3"), 1)
    omega6 = _safe_num(fat.get("omega6"), 1)
    cholesterol = _safe_num(fat.get("cholesterol"), 1)

    # Logical clamps
    if sugar > carbs_total:
        sugar = carbs_total
    if fiber > carbs_total:
        fiber = carbs_total
    if saturated > fat_total:
        saturated = fat_total

    # Net carbs
    net = carbs_total - fiber - sugarAlcohols - allulose
    if net < 0:
        net = 0.0
    net = round(net, 1)

    # Calories via 4/4/9
    calories = _recompute_calories(protein, carbs_total, fat_total)

    # Reassemble
    macros_out = {
        "calories": int(calories),
        "protein": protein,
        "carbohydrates": {
            "total": carbs_total,
            "net": net,
            "fiber": fiber,
            "sugar": sugar,
            "addedSugar": addedSugar,
            "sugarAlcohols": sugarAlcohols,
            "allulose": allulose,
        },
        "fat": {
            "total": fat_total,
            "saturated": saturated,
            "monounsaturated": monounsaturated,
            "polyunsaturated": polyunsaturated,
            "omega3": omega3,
            "omega6": omega6,
            "cholesterol": cholesterol,
        },
    }
    m["macros"] = macros_out

    # Serving normalization: ensure ints for qty/grams, keep unit
    ss = m.get("servingSize") or {}
    qty = ss.get("qty", 1)
    grams = ss.get("grams", 0)
    try:
        qty = int(qty)
    except Exception:
        qty = 1
    try:
        grams = int(float(grams))
    except Exception:
        grams = 0
    unit = ss.get("unit", "plate")
    m["servingSize"] = {"qty": qty, "unit": unit, "grams": grams}

    # Shorten meal name to 2-4 words when too long
    name = (m.get("mealName") or "").strip()
    if name:
        words = name.split()
        if len(words) > 4:
            m["mealName"] = " ".join(words[:4])

    return m

def _normalize_meals(meals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_normalize_meal(meal) for meal in meals]
def _image_part(image_url: str) -> Dict[str, Any]:
    """Builds the image content part for the Chat Completions API."""
    return {"type": "image_url", "image_url": {"url": image_url}}


def _compress_image_to_data_url(image_bytes: bytes, preferred_mime: Optional[str]) -> str:
    """Compress the uploaded image to reduce payload size and return as data URL."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            # Optional resize to reasonable bounds if extremely large
            max_dim = 1600
            w, h = img.size
            if max(w, h) > max_dim:
                scale: float = max_dim / float(max(w, h))
                new_size: tuple = (int(w * scale), int(h * scale))
                img = cast(Any, img).resize(new_size)

            buf = BytesIO()
            # Prefer WEBP if hinted, otherwise JPEG
            use_webp = preferred_mime and preferred_mime.lower() in ("image/webp", "webp")
            fmt = "WEBP" if use_webp else "JPEG"
            save_kwargs: Dict[str, Any] = {"quality": 80}
            if fmt == "WEBP":
                save_kwargs["method"] = 4
            else:
                save_kwargs["optimize"] = True
            img.save(buf, format=fmt, **save_kwargs)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime = "image/webp" if fmt == "WEBP" else "image/jpeg"
            return f"data:{mime};base64,{b64}"
    except Exception:
        # Fallback to original bytes if compression fails
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = preferred_mime or "image/png"
        return f"data:{mime};base64,{b64}"

def analyze_image(image_url: str) -> List[Dict[str, Any]]:
    logger.info("Analyzing image via OpenAI vision model")
    # Cache key includes prompt type, model, image identifier, and determinism flag
    cache_key = f"image::{MODEL}::{DETERMINISTIC}::{hash(image_url)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("Cache hit for analyze_image")
        return cached
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = {
        "model": MODEL,
        "temperature": 0 if DETERMINISTIC else 0.2,
        "top_p": 1,
        "seed": 7 if DETERMINISTIC else None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_IMAGE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                    _image_part(image_url),
                ],
            },
        ],
    }

    resp: Optional[requests.Response] = None
    for attempt, delay in enumerate((0.0, 1.0, 2.0), start=1):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
            break
        except requests.exceptions.RequestException:
            logger.exception("HTTP call to OpenAI failed (attempt %s)", attempt)
            if attempt >= 3:
                raise HTTPException(status_code=502, detail="Upstream API request failed")
            if delay:
                time.sleep(delay)

    if resp is None:
        raise HTTPException(status_code=502, detail="No response from OpenAI after retries")

    logger.info("OpenAI response status: %s", resp.status_code)
    if not resp.ok:
        logger.error("OpenAI API request failed: %s", resp.text[:500])
        raise HTTPException(status_code=resp.status_code, detail=f"OpenAI API request failed: {resp.text}")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        logger.exception("Failed parsing OpenAI JSON response")
        raise HTTPException(status_code=502, detail="Invalid response from OpenAI")
    if content is None:
        refusal_reason = resp.json()["choices"][0]["message"].get("refusal", "Unknown reason")
        logger.error("OpenAI returned no content. Refusal: %s", refusal_reason)
        raise HTTPException(status_code=400, detail=f"API refused to process: {refusal_reason}")

    # Log a short preview of the content for debugging
    logger.info("OpenAI content preview: %s", (content[:200] + ("..." if len(content) > 200 else "")).replace("\n", " "))

    # Use robust extractor
    meals = _extract_meals_from_content(content)
    meals = _normalize_meals(meals)
    _cache_set(cache_key, meals)

    return meals

def analyze_transcription(transcription: str) -> List[Dict[str, Any]]:
    logger.info("Analyzing transcription via OpenAI text model")
    # Normalize whitespace for stable keying
    trans_norm = re.sub(r"\s+", " ", transcription.strip())
    cache_key = f"transcription::{MODEL}::{DETERMINISTIC}::{hash(trans_norm)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("Cache hit for analyze_transcription")
        return cached
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": MODEL,
        "temperature": 0 if DETERMINISTIC else 0.1,
        "top_p": 1,
        "seed": 11 if DETERMINISTIC else None,
        # "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_TRANSCRIPTION},
            {"role": "user", "content": USER_PROMPT_TRANSCRIPTION.replace("{transcription}", transcription)},
        ],
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except Exception:
        logger.exception("HTTP call to OpenAI failed (transcription)")
        raise HTTPException(status_code=502, detail="Upstream API request failed")
    try:
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if content is None:
            refusal_reason = data["choices"][0]["message"].get("refusal", "Unknown reason")
            logger.error("OpenAI returned no content for transcription. Refusal: %s", refusal_reason)
            raise HTTPException(status_code=400, detail=f"API refused to process: {refusal_reason}")

        # Use the same robust extractor as image to handle arrays/wrappers/codefences
        meals = _extract_meals_from_content(content)
        meals = _normalize_meals(meals)
        _cache_set(cache_key, meals)
        return meals
            
    except Exception as e:
        logger.exception("API error or invalid response while analyzing transcription")
        raise HTTPException(status_code=500, detail=f"API error or invalid response: {str(e)}")

def suggest_meal(todays_meals: List[MealTRANSCRIPTION], daily_protein_goal: float) -> List[Dict[str, Any]]:
    """Suggest a meal to help reach daily protein goal. Returns same structure as transcription analysis."""
    logger.info("Generating meal suggestion via OpenAI")
    
    # Calculate totals and remaining protein today
    total_protein_consumed = sum(meal.macros.protein for meal in todays_meals)
    remaining_protein = max(daily_protein_goal - total_protein_consumed, 0)
    
    # Create summary of today's meals
    meals_summary = []
    for i, meal in enumerate(todays_meals, 1):
        meals_summary.append(f"{i}. {meal.mealName} - {meal.macros.protein}g protein, {meal.macros.calories} calories")
    
    todays_meals_text = "\n".join(meals_summary) if meals_summary else "No meals consumed today"
    
    # Get current time context
    from datetime import datetime
    current_hour = datetime.now().hour
    if current_hour < 11:
        time_context = "morning (breakfast/snack time)"
        meals_remaining = 3  # breakfast + lunch + dinner/snack remaining window
    elif current_hour < 15:
        time_context = "afternoon (lunch time)"
        meals_remaining = 2  # lunch + dinner
    elif current_hour < 19:
        time_context = "evening (dinner time)"
        meals_remaining = 1  # dinner
    else:
        time_context = "late evening (snack time)"
        meals_remaining = 1  # snack/late meal

    # Determine balanced mode:
    # Enable when remaining protein is small relative to goal (<= 20% of goal or <= 15g)
    rel_remaining = (remaining_protein / daily_protein_goal) if daily_protein_goal > 0 else 0
    balanced_mode = (remaining_protein <= 15.0) or (rel_remaining <= 0.20)

    # Compute dynamic protein target for this meal
    # If balanced mode, steer toward moderate protein; otherwise distribute remaining across meals
    if balanced_mode:
        # Prefer moderate protein band 10–30g
        target_protein_for_this_meal = 20.0 if meals_remaining <= 1 else 18.0
    else:
        base_target = remaining_protein / max(meals_remaining, 1)
        target_protein_for_this_meal = round(min(max(base_target, 15.0), 60.0), 1)
    # Log computed context for suggestion
    logger.info(
        "SuggestMeal ctx: goal=%s, consumed=%s, remaining=%s, meals_remaining=%s, balanced=%s, target=%s",
        round(daily_protein_goal, 1),
        round(total_protein_consumed, 1),
        round(remaining_protein, 1),
        meals_remaining,
        balanced_mode,
        target_protein_for_this_meal,
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    user_prompt_rendered = USER_PROMPT_MEAL_SUGGESTION.format(
        daily_protein_goal=round(daily_protein_goal, 1),
        total_protein_consumed=round(total_protein_consumed, 1),
        remaining_protein=round(remaining_protein, 1),
        meals_remaining=meals_remaining,
        target_protein_for_this_meal=target_protein_for_this_meal,
        todays_meals_summary=todays_meals_text,
        current_time=time_context,
        balanced_mode=str(balanced_mode).lower()
    )

    # Inject a nonce to promote variety when non-deterministic; model must not echo it
    sm_deterministic = SUGGEST_MEAL_DETERMINISTIC
    if not sm_deterministic:
        nonce = int(time.time() * 1000) % 1_000_000
        user_prompt_rendered += f"\nVARIATION NONCE: {nonce} (do not include this in output)"

    # Cache only when deterministic to ensure repeatability; skip cache for variability
    cache_key = None
    if sm_deterministic:
        cache_key = f"suggest::{MODEL}::{DETERMINISTIC}::{hash(user_prompt_rendered)}"
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.info("Cache hit for suggest_meal")
            return cached

    payload = {
        "model": MODEL,
        "temperature": 0 if sm_deterministic else 0.8,
        "top_p": 1 if sm_deterministic else 0.95,
        "seed": 23 if sm_deterministic else None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_MEAL_SUGGESTION},
            {"role": "user", "content": user_prompt_rendered},
        ],
    }
    logger.info(
        "SuggestMeal params: deterministic=%s, temperature=%s, top_p=%s",
        sm_deterministic,
        payload["temperature"],
        payload["top_p"],
    )

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        
        if content is None:
            refusal_reason = data["choices"][0]["message"].get("refusal", "Unknown reason")
            logger.error("OpenAI returned no content for meal suggestion. Refusal: %s", refusal_reason)
            raise HTTPException(status_code=400, detail=f"API refused to process: {refusal_reason}")

        # Use the same robust extractor as transcription to handle arrays/wrappers/codefences
        meals = _extract_meals_from_content(content)
        meals = _normalize_meals(meals)
        # Log brief summary of first suggested meal
        try:
            first = meals[0] if meals else None
            if first:
                macros = first.get("macros", {})
                logger.info(
                    "SuggestMeal result: %s | protein=%sg | calories=%s",
                    first.get("mealName", "<unknown>"),
                    macros.get("protein"),
                    macros.get("calories"),
                )
        except Exception:
            pass
        if sm_deterministic and cache_key is not None:
            _cache_set(cache_key, meals)
        return meals
            
    except requests.exceptions.RequestException as e:
        logger.exception("HTTP call to OpenAI failed (meal suggestion)")
        raise HTTPException(status_code=502, detail="Upstream API request failed")
    except Exception as e:
        logger.exception("API error while generating meal suggestion")
        raise HTTPException(status_code=500, detail=f"Error generating meal suggestion: {str(e)}")

# -------------------- FastAPI Endpoints --------------------
@app.post("/analyze-image", response_model=List[MealIMAGE])
async def analyze_image_endpoint(request: ImageAnalysisRequest):
    try:
        meals = analyze_image(request.image_url)
        return meals
    except Exception as e:
        logger.exception("Error in /analyze-image endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-transcription", response_model=List[MealTRANSCRIPTION])
async def analyze_transcription_endpoint(request: TranscriptionRequest):
    try:
        meals = analyze_transcription(request.transcription)
        return meals
    except Exception as e:
        logger.exception("Error in /analyze-transcription endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing transcription: {str(e)}")

@app.post("/analyze-image-upload", response_model=List[MealIMAGE])
async def analyze_image_upload(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        data_url = _compress_image_to_data_url(image_bytes, file.content_type)
        meals = analyze_image(data_url)
        return meals
    except Exception as e:
        logger.exception("Error in /analyze-image-upload endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded image: {str(e)}")

@app.post("/suggest-meal", response_model=List[MealTRANSCRIPTION])
async def suggest_meal_endpoint(request: MealSuggestionRequest):
    try:
        meals = suggest_meal(request.todays_meals, request.daily_protein_goal)
        return meals
    except Exception as e:
        logger.exception("Error in /suggest-meal endpoint")
        raise HTTPException(status_code=500, detail=f"Error generating meal suggestion: {str(e)}")
