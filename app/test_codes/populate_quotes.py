import requests
import random

# API endpoint
API_URL = "http://localhost:8000/api/quote"

# Procurement verticals and items
PROCUREMENT_VERTICALS = {
    "manufacturing": ["widgets", "components", "parts", "machinery", "equipment", "tools"],
    "construction": ["materials", "concrete", "steel", "lumber", "pipes", "fixtures"],
    "office": ["furniture", "supplies", "electronics", "computers", "printers"],
    "maintenance": ["cleaning", "repair", "spare parts", "consumables", "chemicals"],
    "it": ["software", "hardware", "servers", "networking", "cables"],
    "logistics": ["packaging", "shipping", "containers", "pallets", "labels"],
    "safety": ["ppe", "helmets", "gloves", "safety equipment", "protective gear"],
    "utilities": ["electrical", "plumbing", "hvac", "lighting", "power"],
    "general": ["miscellaneous item", "general use tool", "standard product", "multi-purpose supply"]
}

SUPPLIERS = [f"Supplier {chr(65 + i)}" for i in range(10)]  # Supplier A to Supplier J

# Generate a random quote entry
def generate_quote(vertical, item):
    quantity = random.randint(1, 100)
    unit_price = round(random.uniform(10.0, 1000.0), 2)
    total_price = round(unit_price * quantity, 2)
    supplier = random.choice(SUPPLIERS)

    content = (
        f"We provide high-quality {item} for {vertical} needs. "
        f"Special rate: ${unit_price:.2f} per unit."
    )

    return {
        "supplier": supplier,
        "content": content,
        "unit_price": unit_price,
        "total_price": total_price,
        "quantity": quantity,
        "vertical": vertical
    }

def populate_quotes(n=200):
    all_quotes = []

    # Ensure at least 4-5 from each vertical
    for vertical, items in PROCUREMENT_VERTICALS.items():
        for _ in range(5):
            item = random.choice(items)
            quote = generate_quote(vertical, item)
            all_quotes.append(quote)

    # Add remaining quotes randomly
    remaining = n - len(all_quotes)
    all_verticals = list(PROCUREMENT_VERTICALS.items())

    for _ in range(remaining):
        vertical, items = random.choice(all_verticals)
        item = random.choice(items)
        quote = generate_quote(vertical, item)
        all_quotes.append(quote)

    # Send quotes to API
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    for i, quote in enumerate(all_quotes, 1):
        response = requests.post(API_URL, json=quote, headers=headers)
        if response.status_code == 200:
            print(f"[{i}/{len(all_quotes)}] Quote added successfully.")
        else:
            print(f"[{i}/{len(all_quotes)}] Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    populate_quotes(200)
