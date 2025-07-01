from langchain.tools import tool
from app.embeddings.faiss_index import search_ids
from app.db.database import SessionLocal
from app.db.models import Quote


@tool
def search_quotes_tool(query: str) -> str:
    """Semantic search over supplier quotes and return top matches."""
    db = SessionLocal()
    try:
        ids = search_ids(query)
        if not ids:
            return "No matching quotes found."

        quotes = db.query(Quote).filter(Quote.id.in_(ids)).all()
        return "\n".join([
            f"Quote ID: {q.id} | Supplier: {q.supplier} | Price: ${q.price:.2f} | Content: {q.content}"
            for q in quotes
        ])
    finally:
        db.close()


@tool
def get_best_quote_under_budget(details: str) -> str:
    """
    Returns the best-priced quote that matches a user need (semantic) and stays within budget.
    Format input as: 'stainless steel rods under 500'
    """
    import re
    db = SessionLocal()
    try:
        match = re.search(r'under\s+\$?(\d+)', details.lower())
        if not match:
            return "Please specify a budget like 'under $500'."

        budget = float(match.group(1))
        ids = search_ids(details)
        if not ids:
            return "No quotes matched your description."

        quotes = db.query(Quote).filter(Quote.id.in_(ids), Quote.price <= budget).order_by(Quote.price.asc()).all()
        if not quotes:
            return "No quotes found under that budget."

        q = quotes[0]
        return f"Best quote under ${budget}:\nSupplier: {q.supplier}, Price: ${q.price:.2f}, Content: {q.content}"
    finally:
        db.close()


@tool
def check_procurement_policy(text: str) -> str:
    """Basic policy checker: Reject quotes that mention 'gold', 'luxury', or price over $10,000."""
    keywords = ["gold", "luxury", "diamond", "platinum"]
    if any(k in text.lower() for k in keywords):
        return "REJECTED: Mentions restricted luxury items."

    import re
    prices = [float(p) for p in re.findall(r"\$?(\d{4,})", text)]
    if any(p > 10000 for p in prices):
        return "REJECTED: Price exceeds $10,000 threshold."

    return "PASSED: Complies with procurement policy."
