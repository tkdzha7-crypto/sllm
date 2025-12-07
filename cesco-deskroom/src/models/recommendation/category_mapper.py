import pandas as pd

from flows.common.db_utils import get_engine

# Cache the mapping in module-level variable
_category_mapping_cache = None


def load_category_mapping():
    """
    Load category mapping from database.
    Uses caching to avoid repeated database queries.
    """
    global _category_mapping_cache

    if _category_mapping_cache is not None:
        return _category_mapping_cache

    engine = get_engine()
    query = "SELECT * FROM source.industry_codes"
    df = pd.read_sql(query, engine)
    _category_mapping_cache = dict(zip(df["code"], df["name"], strict=False))

    return _category_mapping_cache


def get_category_name(code, mapping=None):
    """
    Returns the category name for a given code.

    Args:
        code (str): The category code.
        mapping (dict): Optional mapping dictionary. If None, it loads it.

    Returns:
        str: The category name or "Unknown" if not found.
    """
    if mapping is None:
        mapping = load_category_mapping()
    return mapping.get(code, "Unknown")


if __name__ == "__main__":
    # Test the mapping
    try:
        mapping = load_category_mapping()
        print(f"Successfully loaded {len(mapping)} mappings.")

        # Test a few samples
        test_codes = ["A01110", "A01121", "C10111"]
        for code in test_codes:
            print(f"{code} -> {get_category_name(code, mapping)}")

    except Exception as e:
        print(f"Failed to load mapping: {e}")
