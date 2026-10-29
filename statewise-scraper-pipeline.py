import os
import time
import pandas as pd
from apify_client import ApifyClient

# Configuration
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "YOUR_APIFY_API_TOKEN")
BATCH_SIZE = 200          # Number of ZIP codes per Apify request batch
COOLDOWN_SECONDS = 20     # Pause between batches to avoid platform fatigue

client = ApifyClient(APIFY_TOKEN)

# USPS ZIP code prefix ranges mapped to state abbreviations.
# Each entry is: (state_abbreviation, min_prefix, max_prefix)
ZIP_PREFIX_RANGES = [
    ("AK", 995, 999),
    ("AL", 350, 369),
    ("AR", 716, 729),
    ("AZ", 850, 865),
    ("CA", 900, 961),
    ("CO", 800, 816),
    ("CT", 60, 69),
    ("DC", 200, 205),
    ("DE", 197, 199),
    ("FL", 320, 349),
    ("GA", 300, 319),
    ("HI", 967, 968),
    ("IA", 500, 528),
    ("ID", 832, 838),
    ("IL", 600, 629),
    ("IN", 460, 479),
    ("KS", 660, 679),
    ("KY", 400, 427),
    ("LA", 700, 714),
    ("MA", 10, 27),
    ("MD", 210, 219),
    ("ME", 39, 49),
    ("MI", 480, 499),
    ("MN", 550, 567),
    ("MO", 630, 658),
    ("MS", 386, 397),
    ("MT", 590, 599),
    ("NC", 270, 289),
    ("ND", 580, 588),
    ("NE", 680, 693),
    ("NH", 30, 38),
    ("NJ", 70, 89),
    ("NM", 870, 884),
    ("NV", 889, 898),
    ("NY", 100, 149),
    ("OH", 430, 459),
    ("OK", 730, 749),
    ("OR", 970, 979),
    ("PA", 150, 196),
    ("RI", 28, 29),
    ("SC", 290, 299),
    ("SD", 570, 577),
    ("TN", 370, 385),
    ("TX", 750, 799),
    ("UT", 840, 847),
    ("VA", 220, 246),
    ("VT", 50, 59),
    ("WA", 980, 994),
    ("WI", 530, 549),
    ("WV", 247, 268),
    ("WY", 820, 831),
]


def get_state_from_zip(zip_code):
    """
    Approximate the US state from a 5-digit ZIP code prefix.
    This avoids needing a heavy database for basic geographic filtering.
    """
    try:
        # Pad with leading zeros if needed, then take the first 3 digits.
        zip_str = str(zip_code).zfill(5)[:3]
        zip_prefix = int(zip_str)
    except ValueError:
        return "UNKNOWN"

    for state, min_prefix, max_prefix in ZIP_PREFIX_RANGES:
        if min_prefix <= zip_prefix <= max_prefix:
            return state

    return "UNKNOWN"


def load_and_filter_zips_by_state(file_path, target_states=None):
    """
    Load ZIP codes from a file and optionally filter them by target state(s).

    target_states can be:
      - None: return all ZIP codes
      - a string, e.g. "CA"
      - a list, e.g. ["NY", "NJ", "CT"]
    """
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please provide your source ZIP codes.")
        return []

    raw_zips = []

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, dtype=str)
            zip_cols = [
                col
                for col in df.columns
                if "zip" in col.lower() or "postal" in col.lower()
            ]
            if zip_cols:
                raw_zips = df[zip_cols].dropna().iloc[:, 0].tolist()
            else:
                raw_zips = df.iloc[:, 0].dropna().tolist()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_zips = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading source file: {e}")
        return []

    # No state filter requested: return all ZIP codes normalized to 5 digits.
    if not target_states:
        return [str(z).zfill(5) for z in raw_zips]

    # Normalize target states to a list of uppercase abbreviations.
    if isinstance(target_states, str):
        target_states = [target_states.upper()]
    else:
        target_states = [state.upper() for state in target_states]

    filtered_zips = []
    for z in raw_zips:
        padded_zip = str(z).zfill(5)
        if get_state_from_zip(padded_zip) in target_states:
            filtered_zips.append(padded_zip)

    return filtered_zips


def scrape_batch(zip_batch, business_type, min_rating, min_reviews):
    """Execute a single Apify scrape batch."""
    search_queries = [f"{business_type} {z}" for z in zip_batch]

    run_input = {
        "queries": search_queries,
        "searchMode": "target_qualified",
        "minRating": min_rating,
        "minReviews": min_reviews,
        "excludeClosed": True,
        "maxPlacesPerQuery": 50,
    }

    try:
        run = client.actor(
            "code-node-tools/google-maps-businesses-without-websites"
        ).call(run_input=run_input)

        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

        extracted_leads = []
        for item in dataset_items:
            if not item.get("website"):
                extracted_leads.append(
                    {
                        "Name": item.get("title") or item.get("name"),
                        "Category": item.get("categoryName"),
                        "Phone": item.get("phone")
                        or item.get("formattedPhone")
                        or "N/A",
                        "Address": item.get("address"),
                        "Rating": item.get("reviewRating") or item.get("rating"),
                        "Reviews Count": item.get("reviewCount")
                        or item.get("userRatingsTotal"),
                        "ZIP Code": item.get("postalCode"),
                    }
                )

        return extracted_leads

    except Exception as e:
        print(f" [Warning] Error in current batch run: {e}")
        return []


def append_to_file(new_leads, output_path):
    """Append records to a file and remove duplicates where possible."""
    if not new_leads:
        return

    new_df = pd.DataFrame(new_leads)

    if output_path.endswith(".xlsx"):
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_excel(output_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(
                    subset=["Name", "Phone"],
                    keep="first",
                    inplace=True,
                )
                combined_df.to_excel(output_path, index=False)
            except Exception:
                new_df.to_excel(output_path, index=False)
        else:
            new_df.to_excel(output_path, index=False)
    else:
        file_exists = os.path.exists(output_path)
        new_df.to_csv(
            output_path,
            mode="a",
            index=False,
            header=not file_exists,
            encoding="utf-8",
        )


def main():
    # -----------------------------------------------------------------------
    # Workflow configuration
    # -----------------------------------------------------------------------
    ZIP_INPUT_FILE = "us_zip_codes.txt"       # Master input file
    BUSINESS_KEYWORD = "Roofing Contractor"

    # Geographic filter settings.
    # Use a string like "CA" or a list like ["NY", "NJ", "CT"].
    TARGET_STATES = "CA"

    # Output file is named per state, or "multi_state" when filtering multiple states.
    state_label = (
        TARGET_STATES.lower()
        if isinstance(TARGET_STATES, str)
        else "multi_state"
    )
    LEADS_OUTPUT_FILE = f"leads_{state_label}.xlsx"

    if APIFY_TOKEN == "YOUR_APIFY_API_TOKEN":
        print("[Abort]: Please configure your APIFY_API_TOKEN before running.")
        return

    # Load and isolate only the ZIP codes within the target state boundaries.
    state_zips = load_and_filter_zips_by_state(
        ZIP_INPUT_FILE,
        target_states=TARGET_STATES,
    )
    total_zips = len(state_zips)

    if total_zips == 0:
        print(
            f"No ZIP codes found matching the filter: {TARGET_STATES}. "
            "Verify your source file data."
        )
        return

    print(
        f"Geographic Filter Active: Isolated {total_zips} ZIP codes "
        f"for State(s): {TARGET_STATES}"
    )
    print(f"Targeting Market Niche: '{BUSINESS_KEYWORD}'")

    # Process ZIP codes in batches.
    for i in range(0, total_zips, BATCH_SIZE):
        current_batch = state_zips[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_zips + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"\n--- Processing Batch {batch_num}/{total_batches} "
            f"({len(current_batch)} ZIPs in {TARGET_STATES}) ---"
        )

        batch_results = scrape_batch(
            zip_batch=current_batch,
            business_type=BUSINESS_KEYWORD,
            min_rating=4.5,
            min_reviews=10,
        )

        print(f" -> Extracted {len(batch_results)} highly-rated, website-free leads.")
        append_to_file(batch_results, LEADS_OUTPUT_FILE)

        if i + BATCH_SIZE < total_zips:
            print(
                f" -> Cooldown activated. Resting {COOLDOWN_SECONDS} seconds "
                "before the next batch..."
            )
            time.sleep(COOLDOWN_SECONDS)

    print(
        f"\n[Finished] Extraction for state profile {TARGET_STATES} "
        "completed successfully."
    )
    print(f"Results saved to: {LEADS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
