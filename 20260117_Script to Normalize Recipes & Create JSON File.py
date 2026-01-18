#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import json
import pandas as pd
from pathlib import Path


# In[ ]:


# =============================
# CONFIG
# =============================
XLSX_PATH = "/mnt/data/Meal Agent Recipes.xlsx"
OUT_JSON_PATH = "/mnt/data/recipes.json"

# =============================
# HELPERS
# =============================

def normalize_key(k):
    if k is None or (isinstance(k, float) and pd.isna(k)):
        return ""
    s = str(k).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def safe_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()

# --- Ingredient parsing helpers ---
FRACTION_MAP = {
    "¼":"1/4","½":"1/2","¾":"3/4","⅓":"1/3","⅔":"2/3",
    "⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8"
}

UNIT_ALIASES = {
    "tsp":"tsp","teaspoon":"tsp","teaspoons":"tsp",
    "tbsp":"tbsp","tablespoon":"tbsp","tablespoons":"tbsp",
    "cup":"cup","cups":"cup",
    "oz":"oz","ounce":"oz","ounces":"oz",
    "lb":"lb","lbs":"lb","pound":"lb","pounds":"lb",
    "g":"g","gram":"g","grams":"g",
    "ml":"ml","milliliter":"ml","milliliters":"ml",
    "l":"l","liter":"l","liters":"l",
    "clove":"clove","cloves":"clove",
    "can":"can","cans":"can",
    "pinch":"pinch","dash":"dash"
}

SECTION_HEADERS = {
    "crust","pie crust","filling","pie filling",
    "sauce","dressing","marinade","glaze",
    "topping","toppings","garnish","assembly",
    "to serve","for serving"
}

def normalize_fractions(s):
    for k, v in FRACTION_MAP.items():
        s = s.replace(k, v)
    return s

def parse_quantity(q):
    if not q:
        return None
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", q)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.match(r"^(\d+)/(\d+)$", q)
    if m:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(q)
    except:
        return None

def parse_ingredient_line(line):
    raw = line.strip()
    s = normalize_fractions(raw)

    qty = None
    rest = s

    m = re.match(r"^(\d+\s+\d+/\d+|\d+/\d+|\d+(\.\d+)?)\s+(.*)$", s)
    if m:
        qty = parse_quantity(m.group(1))
        rest = m.group(4)

    tokens = rest.split()
    unit = ""
    item = rest

    if tokens and tokens[0].lower().rstrip(".") in UNIT_ALIASES:
        unit = UNIT_ALIASES[tokens[0].lower().rstrip(".")]
        item = " ".join(tokens[1:])

    return {
        "raw": raw,
        "quantity": qty,
        "unit": unit,
        "item": item.strip()
    }

def split_ingredients(cell):
    if pd.isna(cell):
        return []
    lines = [l.strip() for l in str(cell).split("\n") if l.strip()]
    out = []
    for l in lines:
        if l.lower().strip(":") in SECTION_HEADERS:
            continue
        out.append(parse_ingredient_line(l))
    return out

def split_instructions(cell):
    if pd.isna(cell):
        return []
    text = str(cell).replace("\n", " ").strip()
    steps = re.split(r"\s*\d+[\.\)]\s*", text)
    return [s.strip() for s in steps if s.strip()]

def infer_meal_type(title, link):
    t = (title or "").lower()
    u = (link or "").lower()

    if any(k in u for k in ["dessert","cake","cookie","brownie","pie"]):
        return "dessert"
    if any(k in u for k in ["breakfast","brunch"]):
        return "breakfast"
    if any(k in u for k in ["snack","side","dip","appetizer"]):
        return "snack/side dish"

    if any(k in t for k in ["cake","cookie","brownie","dessert","pie"]):
        return "dessert"
    if any(k in t for k in ["pancake","waffle","oatmeal","smoothie"]):
        return "breakfast"
    if any(k in t for k in ["dip","salad","slaw","side"]):
        return "snack/side dish"

    return "lunch/dinner"

# =============================
# MAIN
# =============================
sheets = pd.read_excel(XLSX_PATH, sheet_name=None, header=None)
recipes = []

for sheet_name, df in sheets.items():
    df = df.dropna(how="all")
    if df.shape[1] < 2:
        continue

    kv = {}
    for k, v in zip(df.iloc[:,0], df.iloc[:,1]):
        key = normalize_key(k)
        if key:
            kv[key] = v

    if "title" not in kv or "ingredients" not in kv:
        continue

    title = safe_str(kv.get("title")) or sheet_name
    link = safe_str(kv.get("link"))

    recipe = {
        "source_sheet": sheet_name,
        "title": title,
        "meal_type": infer_meal_type(title, link),
        "ingredients": split_ingredients(kv.get("ingredients")),
        "instructions": split_instructions(kv.get("instructions")),
        "instructions_raw": safe_str(kv.get("instructions")),
        "link": link
    }

    # include all other fields as raw strings
    for k, v in kv.items():
        if k not in recipe:
            recipe[k] = safe_str(v)

    recipes.append(recipe)

with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(recipes, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(recipes)} recipes → {OUT_JSON_PATH}")

