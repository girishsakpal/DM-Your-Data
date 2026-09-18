"""
Tests the CSV upload pipeline end-to-end without needing the browser file
picker. Generates a small sample CSV (with one long-text column, so text
detection has something to find), runs it through replace_active_dataset(),
and prints the resulting metadata.

Usage:
    python scripts/test_upload.py            # uses the built-in sample CSV
    python scripts/test_upload.py path.csv    # uses your own CSV file
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_upload.ingest import replace_active_dataset, UploadError
from app.data_upload import registry

SAMPLE_CSV = """restaurant_name,cuisine,price_range,rating,city,feedback
Blue Ginger,Thai,2,4.5,Portland,"The pad thai was excellent but service was slow on a busy Friday night."
Casa Roma,Italian,3,3.8,Portland,"Decent pasta, nothing special. Would not rush back but wouldn't avoid it either."
Sunset Grill,American,2,4.9,Seattle,"Best burger I've had in years, the staff remembered our order from last time."
Green Bowl,Vegan,1,2.1,Seattle,"Arrived cold and the portion was tiny for the price. Very disappointing visit."
Ocean Breeze,Seafood,4,4.6,Portland,"Fresh oysters and a stunning view, worth every penny for a special occasion."
"""


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(path)
    else:
        print("No CSV path given - using the built-in sample restaurant dataset.\n")
        file_bytes = SAMPLE_CSV.encode("utf-8")
        filename = "sample_restaurants.csv"

    try:
        metadata = replace_active_dataset(file_bytes, filename)
    except UploadError as e:
        print(f"Upload rejected: {e}")
        return

    print("Upload succeeded.")
    print(f"  Table:            {metadata['table']}")
    print(f"  Row count:        {metadata['row_count']}")
    print(f"  Columns:          {metadata['columns']}")
    print(f"  Text column:      {metadata['text_column'] or '(none detected - SQL only)'}")
    print(f"  Embedded rows:    {metadata['embedded_row_count']}")

    print("\nTry it now:")
    print('  python scripts/test_phase1.py "your question about this new data"')
    if metadata["text_column"]:
        print('  python scripts/test_phase2.py "your semantic query about this new data"')

    print("\nTo go back to the seed product_reviews data:")
    print("  python -c \"from app.data_upload.registry import clear_active_dataset; clear_active_dataset()\"")


if __name__ == "__main__":
    main()
