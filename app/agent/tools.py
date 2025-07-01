## app/agent/tools.py
from app.db.database import SessionLocal
from app.db.models import Quote
from app.embeddings.chroma_index import search_quote_ids
from langchain.tools import tool
from sqlalchemy.orm import Session
import os
import requests
import re
import spacy
from typing import Dict, List, Optional
from datetime import datetime

PROCUREMENT_VERTICALS = {
    "manufacturing": ["widgets", "components", "parts", "machinery", "equipment", "tools"],
    "construction": ["materials", "concrete", "steel", "lumber", "pipes", "fixtures"],
    "office": ["furniture", "supplies", "electronics", "computers", "printers"],
    "maintenance": ["cleaning", "repair", "spare parts", "consumables", "chemicals"],
    "it": ["software", "hardware", "servers", "networking", "cables"],
    "logistics": ["packaging", "shipping", "containers", "pallets", "labels"],
    "safety": ["ppe", "helmets", "gloves", "safety equipment", "protective gear"],
    "utilities": ["electrical", "plumbing", "hvac", "lighting", "power"]
}

# Load spacy model (you'll need to install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")

def detect_vertical(query: str) -> str:
    """Detect procurement vertical from query."""
    query_lower = query.lower()
    
    for vertical, keywords in PROCUREMENT_VERTICALS.items():
        if any(keyword in query_lower for keyword in keywords):
            return vertical
    
    return "general"

def extract_preferences(text: str) -> Dict:
    """Enhanced preference extraction using NLP and pattern matching."""
    result = {}
    text_lower = text.lower()
    
    # Enhanced budget extraction
    budget_patterns = [
        r"(?:under|below|less\s+than|budget\s+of|maximum|up\s+to|max|not\s+more\s+than)\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",
        r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:or\s+less|maximum|max|budget)",
        r"budget[:=]\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)"
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, text_lower)
        if match:
            budget_str = match.group(1).replace(',', '')
            result["budget"] = float(budget_str)
            break
    
    quantity_patterns = [
        r"(?:minimum|at\s+least|quantity\s+of|qty\s+of|need\s+at\s+least)\s+(\d+)",
        r"(\d+)\s*(?:units?|pieces?|items?|pcs?)\s*(?:or\s+more|minimum|min)",
        r"qty[:=]\s*(\d+)"
    ]
    
    for pattern in quantity_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["min_quantity"] = int(match.group(1))
            break

    result["vertical"] = detect_vertical(text)


    quality_terms = {"premium": 3, "high quality": 2, "standard": 1, "basic": 0}
    for term, level in quality_terms.items():
        if term in text_lower:
            result["quality_level"] = level
            break

    negotiable_keywords = ["negotiable", "flexible", "discuss", "open to offers"]
    if any(keyword in text_lower for keyword in negotiable_keywords):
        result["negotiable"] = True
    
    # Extract material preferences
    materials = ["steel", "aluminum", "plastic", "wood", "ceramic", "glass", "rubber", "copper", "brass"]
    found_materials = [mat for mat in materials if mat in text_lower]
    if found_materials:
        result["materials"] = found_materials
    
    # Extract size/dimension info
    size_pattern = r"(\d+(?:\.\d+)?)\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:x|by|\*)?\s*(\d+(?:\.\d+)?)?\s*(mm|cm|m|inch|inches|ft|feet)?"
    size_match = re.search(size_pattern, text_lower)
    if size_match:
        dimensions = [size_match.group(1), size_match.group(2)]
        if size_match.group(3):
            dimensions.append(size_match.group(3))
        result["dimensions"] = dimensions
        if size_match.group(4):
            result["dimension_unit"] = size_match.group(4)
    
    # Extract urgency
    urgent_keywords = ["urgent", "asap", "immediately", "rush", "emergency"]
    if any(keyword in text_lower for keyword in urgent_keywords):
        result["urgent"] = True
    
    # Extract quality requirements
    quality_keywords = ["high quality", "premium", "standard", "basic", "industrial grade"]
    for quality in quality_keywords:
        if quality in text_lower:
            result["quality"] = quality
            break
    
    return result


def expand_search_terms(query: str) -> List[str]:
    """Generate multiple search variations for better matching."""
    variations = [query]
    
    # Add synonyms and related terms
    synonyms_map = {
        "widget": ["component", "part", "device", "unit", "element"],
        "industrial": ["commercial", "manufacturing", "factory", "production"],
        "steel": ["metal", "iron", "alloy"],
        "aluminum": ["aluminium", "metal", "alloy"],
        "pipe": ["tube", "conduit", "piping"],
        "bolt": ["screw", "fastener", "hardware"],
        "cable": ["wire", "cord", "line"],
        "motor": ["engine", "drive", "actuator"]
    }
    
    query_lower = query.lower()
    for word, syns in synonyms_map.items():
        if word in query_lower:
            for syn in syns:
                variations.append(query_lower.replace(word, syn))
    
    # Add partial matches
    words = query.split()
    if len(words) > 1:
        # Try individual words
        variations.extend(words)
        # Try pairs of words
        for i in range(len(words) - 1):
            variations.append(f"{words[i]} {words[i+1]}")
    
    return list(set(variations))  # Remove duplicates


def semantic_search_fallback(query: str, all_quotes: List) -> List:
    """Use NLP similarity when exact matching fails."""
    if not nlp or not all_quotes:
        return []
    
    query_doc = nlp(query)
    scored_quotes = []
    
    for quote in all_quotes:
        quote_doc = nlp(quote.content)
        similarity = query_doc.similarity(quote_doc)
        if similarity > 0.3:  # Minimum similarity threshold
            scored_quotes.append((quote, similarity))
    
    # Sort by similarity score
    scored_quotes.sort(key=lambda x: x[1], reverse=True)
    return [quote for quote, score in scored_quotes[:10]]


def calculate_best_quote(quotes: List[Quote], quantity: int, budget: Optional[float] = None) -> Dict:
    """Calculate best quote considering quantity and total cost."""
    viable_quotes = []
    
    for quote in quotes:
        # Check if supplier can meet quantity
        if quote.quantity and quantity < quote.quantity:
            continue  # Skip if we need less than minimum quantity
        
        # Calculate actual costs
        if quote.unit_price:
            total_cost = quote.unit_price * quantity
        else:
            total_cost = quote.total_price or 0
        
        # Check budget constraint
        if budget and total_cost > budget:
            continue
        
        viable_quotes.append({
            "quote": quote,
            "total_cost": total_cost,
            "unit_cost": total_cost / quantity if quantity > 0 else quote.unit_price or 0
        })
    
    if not viable_quotes:
        return {"viable": False, "alternatives": quotes[:3]}
    
    viable_quotes.sort(key=lambda x: x["total_cost"])
    return {"viable": True, "best": viable_quotes[0], "alternatives": viable_quotes[1:3]}


def llm_rerank_fallback(text: str, quote_list: list[Quote]) -> str:
    """Call Hugging Face LLM to choose the best among quote_list based on text description."""
    prompt = (
        f"You are a helpful procurement assistant.\n"
        f"User request: {text}\n"
        f"Available quotes:\n"
    )
    for i, q in enumerate(quote_list):
        prompt += (
            f"{i+1}. Supplier: {q.supplier}, Price: ${q.price:.2f}, "
            f"Details: {q.content}\n"
        )
    prompt += "Pick the best quote and explain why."

    response = requests.post(
        "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        headers={"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    )

    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Could not evaluate fallback ranking. Please try again."



@tool
def search_quotes_by_vertical(query: str) -> str:
    """Search quotes by procurement vertical with enhanced filtering."""
    db = SessionLocal()
    try:
        prefs = extract_preferences(query)
        vertical = prefs.get("vertical", "general")
        
        # Search by vertical first
        base_query = db.query(Quote)
        if vertical != "general":
            base_query = base_query.filter(Quote.vertical == vertical)
        
        # Then apply semantic search
        search_variations = expand_search_terms(query)
        all_matching_ids = set()
        
        for variation in search_variations:
            ids = search_quote_ids(variation, top_k=10, threshold=0.4)
            all_matching_ids.update(ids)
        
        if all_matching_ids:
            quotes = base_query.filter(Quote.id.in_(all_matching_ids)).all()
        else:
            # Fallback to vertical-only search
            quotes = base_query.limit(10).all()
        
        if not quotes:
            return f"No quotes found for {vertical} vertical. Available verticals: {', '.join(PROCUREMENT_VERTICALS.keys())}"
        
        result = f"Found {len(quotes)} quotes in {vertical} vertical:\n\n"
        for i, q in enumerate(quotes, 1):
            unit_price = f"${q.unit_price:.2f}/unit" if q.unit_price else "N/A"
            total_price = f"${q.total_price:.2f} total" if q.total_price else "N/A"
            qty_info = f"Min qty: {q.quantity}" if q.quantity else "No min qty"
            
            result += f"{i}. {q.supplier} | {unit_price} | {total_price} | {qty_info}\n"
            result += f"   {q.content}\n\n"
        
        return result.strip()
    
    finally:
        db.close()

@tool
def find_best_quote_for_quantity(query: str) -> str:
    """Find optimal quote considering quantity requirements and total cost."""
    db = SessionLocal()
    try:
        prefs = extract_preferences(query)
        quantity = prefs.get("quantity", 1)
        budget = prefs.get("budget")
        vertical = prefs.get("vertical", "general")
        
        # Search for matching quotes
        search_variations = expand_search_terms(query)
        all_matching_ids = set()
        
        for variation in search_variations:
            ids = search_quote_ids(variation, top_k=15, threshold=0.4)
            all_matching_ids.update(ids)
        
        if all_matching_ids:
            quotes = db.query(Quote).filter(Quote.id.in_(all_matching_ids)).all()
        else:
            # Fallback to vertical search
            if vertical != "general":
                quotes = db.query(Quote).filter(Quote.vertical == vertical).all()
            else:
                quotes = db.query(Quote).limit(20).all()
        
        if not quotes:
            return "No matching quotes found. Please add quotes to the database first."
        
        # Calculate best option
        result = calculate_best_quote(quotes, quantity, budget)
        
        if result["viable"]:
            best = result["best"]
            quote = best["quote"]
            
            response = f"Best quote for {quantity} units:\n"
            response += f"Supplier: {quote.supplier}\n"
            response += f"Unit Price: ${best['unit_cost']:.2f}\n"
            response += f"Total Cost: ${best['total_cost']:.2f}\n"
            response += f"Details: {quote.content}\n"
            
            if budget:
                savings = budget - best['total_cost']
                response += f"Budget: ${budget:.2f} (${savings:.2f} under budget)\n"
            
            # Show alternatives
            if result["alternatives"]:
                response += f"\nAlternatives:\n"
                for alt in result["alternatives"][:2]:
                    alt_quote = alt["quote"]
                    response += f"- {alt_quote.supplier}: ${alt['total_cost']:.2f} total\n"
            
            return response
        
        else:
            # No viable options
            response = f"No quotes can meet requirement for {quantity} units"
            if budget:
                response += f" under ${budget:.2f}"
            response += ".\n\nClosest alternatives:\n"
            
            for quote in result["alternatives"][:3]:
                if quote.unit_price:
                    estimated_cost = quote.unit_price * quantity
                    response += f"- {quote.supplier}: ${estimated_cost:.2f} estimated total"
                    if quote.quantity and quantity < quote.quantity:
                        response += f" (min qty: {quote.quantity})"
                    response += "\n"
            
            return response
    
    finally:
        db.close()

@tool
def procurement_requirements_form(query: str) -> str:
    """Generate a requirements form when query lacks specific details."""
    prefs = extract_preferences(query)
    
    form = "PROCUREMENT REQUIREMENTS FORM\n"
    form += "=" * 35 + "\n\n"
    
    # What we detected
    if prefs:
        form += "Detected Requirements:\n"
        if "quantity" in prefs:
            form += f"✓ Quantity: {prefs['quantity']} units\n"
        if "budget" in prefs:
            form += f"✓ Budget: ${prefs['budget']:.2f}\n"
        if "vertical" in prefs:
            form += f"✓ Category: {prefs['vertical']}\n"
        form += "\n"
    
    # Missing information
    form += "Please provide missing details:\n\n"
    
    if "quantity" not in prefs:
        form += "□ Quantity needed: _____ units\n"
    
    if "budget" not in prefs:
        form += "□ Budget range: $_____ to $_____\n"
    
    form += "□ Delivery timeline: _____ days/weeks\n"
    form += "□ Quality requirements: □ Basic □ Standard □ Premium\n"
    form += "□ Technical specifications: _________________\n"
    form += "□ Preferred suppliers (if any): _____________\n"
    form += "□ Special requirements: ____________________\n\n"
    
    form += "Available Categories:\n"
    for vertical, keywords in PROCUREMENT_VERTICALS.items():
        form += f"• {vertical.title()}: {', '.join(keywords[:3])}\n"
    
    form += "\nProvide complete details for better quote matching."
    
    return form

@tool
def search_best_quote(details: str) -> str:
    """Enhanced quote search with multiple strategies and fallbacks."""
    db: Session = SessionLocal()
    try:
        prefs = extract_preferences(details)
        budget = prefs.get("budget", None)
        
        # Strategy 1: Try multiple search variations
        search_variations = expand_search_terms(details)
        all_matching_ids = set()
        
        for variation in search_variations:
            ids = search_quote_ids(variation, top_k=10, threshold=0.5)  # Lower threshold
            all_matching_ids.update(ids)
        
        # Strategy 2: If no semantic matches, try keyword-based search
        if not all_matching_ids:
            keywords = details.lower().split()
            db_quotes = db.query(Quote).all()
            
            for quote in db_quotes:
                content_lower = quote.content.lower()
                if any(keyword in content_lower for keyword in keywords):
                    all_matching_ids.add(quote.id)
        
        # Strategy 3: If still no matches, use NLP similarity
        if not all_matching_ids:
            all_quotes = db.query(Quote).all()
            similar_quotes = semantic_search_fallback(details, all_quotes)
            all_matching_ids.update(q.id for q in similar_quotes)
        
        # Get quotes from IDs
        if all_matching_ids:
            quotes = db.query(Quote).filter(Quote.id.in_(all_matching_ids)).order_by(Quote.total_price.asc()).all()
        else:
            # Last resort: return cheapest quotes
            quotes = db.query(Quote).order_by(Quote.price.asc()).limit(5).all()
        
        if not quotes:
            return "No supplier quotes found in the database. Please add some quotes first."
        
        # Apply budget filter
        if budget:
            budget_filtered = [q for q in quotes if q.price <= budget]
            if budget_filtered:
                best = budget_filtered[0]
                return (
                    f"Best quote under ${budget}:\n"
                    f"Supplier: {best.supplier}\n"
                    f"Price: ${best.price:.2f}\n"
                    f"Details: {best.content}\n"
                    f"Match confidence: High"
                )
            else:
                # Show closest alternatives
                closest = min(quotes, key=lambda q: abs(q.price - budget))
                return (
                    f"No quotes found under ${budget}. Closest option:\n"
                    f"Supplier: {closest.supplier}\n"
                    f"Price: ${closest.price:.2f} (${closest.price - budget:.2f} over budget)\n"
                    f"Details: {closest.content}"
                )
        else:
            # No budget constraint
            best = quotes[0]
            return (
                f"Best matching quote:\n"
                f"Supplier: {best.supplier}\n"
                f"Price: ${best.price:.2f}\n"
                f"Details: {best.content}\n"
                f"Found {len(quotes)} total matches"
            )
    
    finally:
        db.close()


@tool
def search_quotes_tool(query: str) -> str:
    """Enhanced semantic search with multiple fallback strategies."""
    db = SessionLocal()
    try:
        # Strategy 1: Multiple search variations
        search_variations = expand_search_terms(query)
        all_matching_ids = set()
        
        for variation in search_variations:
            ids = search_quote_ids(variation, top_k=8, threshold=0.4)
            all_matching_ids.update(ids)
        
        # Strategy 2: Keyword-based fallback
        if not all_matching_ids:
            keywords = query.lower().split()
            db_quotes = db.query(Quote).all()
            
            for quote in db_quotes:
                content_lower = quote.content.lower()
                supplier_lower = quote.supplier.lower()
                # Check content and supplier name
                if any(keyword in content_lower or keyword in supplier_lower for keyword in keywords):
                    all_matching_ids.add(quote.id)
        
        # Strategy 3: NLP similarity fallback
        if not all_matching_ids:
            all_quotes = db.query(Quote).all()
            similar_quotes = semantic_search_fallback(query, all_quotes)
            all_matching_ids.update(q.id for q in similar_quotes)
        
        if not all_matching_ids:
            # Show available categories/types
            all_quotes = db.query(Quote).limit(20).all()
            if all_quotes:
                sample_items = list(set([q.content.split()[0] for q in all_quotes[:10]]))[:5]
                return f"No matches found for '{query}'. Available items include: {', '.join(sample_items)}"
            else:
                return "No quotes found in database. Please add some quotes first."
        
        quotes = db.query(Quote).filter(Quote.id.in_(all_matching_ids)).order_by(Quote.total_price.asc()).limit(8).all()
        
        result = f"Found {len(quotes)} matches for '{query}':\n\n"
        for i, q in enumerate(quotes, 1):
            result += f"{i}. Supplier: {q.supplier} | Price: ${q.total_price:.2f}\n   Details: {q.content}\n\n"
        
        return result.strip()
        
    finally:
        db.close()



@tool
def check_procurement_policy(text: str) -> str:
    """Enhanced policy checker with quantity and vertical considerations."""
    # Extract pricing and quantity info
    prefs = extract_preferences(text)
    vertical = prefs.get("vertical", "general")
    
    # Vertical-specific limits
    vertical_limits = {
        "office": 5000,
        "it": 15000, 
        "construction": 25000,
        "manufacturing": 50000,
        "general": 10000
    }
    
    limit = vertical_limits.get(vertical, 10000)
    
    # Check for restricted items
    restricted = ["gold", "luxury", "diamond", "platinum", "personal"]
    if any(item in text.lower() for item in restricted):
        return f"REJECTED: Contains restricted items for {vertical} procurement."
    
    # Check budget limits
    if "budget" in prefs and prefs["budget"] > limit:
        return f"REJECTED: Budget ${prefs['budget']:.2f} exceeds {vertical} limit of ${limit:,.2f}"
    
    # Check quantity reasonableness
    if "quantity" in prefs and prefs["quantity"] > 10000:
        return "FLAGGED: Large quantity order requires additional approval."
    
    return f"PASSED: Complies with {vertical} procurement policy (limit: ${limit:,.2f})"
