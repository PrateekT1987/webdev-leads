import os
import re
import time

import pandas as pd
import requests

# Configuration constants
SCRAPERAPI_API_KEY = os.getenv("SCRAPERAPI_API_KEY", "7969ccca4d264055419af72f2c848e00")
BATCH_SIZE = 200  # Split requests into sets of 200 ZIP codes per batch
COOLDOWN_SECONDS = 20  # Wait time between batches to prevent platform fatigue

SCRAPERAPI_SERP_ENDPOINT = "https://api.scraperapi.com/structured/google/search"
SCRAPERAPI_MAPS_ENDPOINT = "https://api.scraperapi.com/structured/google/mapssearch"
_MAPS_UNAVAILABLE = False  # flip to True after the first blocked Maps call


class MapsUnavailable(Exception):
    """Raised when the structured Maps endpoint is blocked or errors out."""

# NOTE: Geo-targeting on the free tier is limited.
#   - The structured Google Maps endpoint (/structured/google/mapssearch) is a
#     ScraperAPI "protected domain": on free keys it returns HTTP 500, but it is
#     the ideal source of per-ZIP businesses WITHOUT websites. If this run ever
#     succeeds on /structured/google/mapssearch (i.e. after upgrading to a paid
#     plan), it is used automatically for every ZIP. Otherwise we fall back to
#     Google Search SERPs, which do work: "local_packs" may be US-wide top-rated
#     rather than strictly ZIP-localized, but "organic_results" for a
#     "<keyword> <ZIP>" query are returned for that area.


def get_state_from_zip(zip_code):
    """
    Approximates the US State from a 5-digit ZIP code prefix.
    Ensures complete architectural isolation without requiring heavy databases.
    """
    try:
        # Pad with leading zeros if it arrived as an integer/short string
        zip_str = str(zip_code).zfill(5)[:3]
        zip_3 = int(zip_str)
    except ValueError:
        return "UNKNOWN"

    # Standard USPS ZIP Code Prefix Mapping Ranges
    if 995 <= zip_3 <= 999:
        return "AK"
    if 350 <= zip_3 <= 369:
        return "AL"
    if 716 <= zip_3 <= 729:
        return "AR"
    if 850 <= zip_3 <= 865:
        return "AZ"
    if 900 <= zip_3 <= 961:
        return "CA"
    if 800 <= zip_3 <= 816:
        return "CO"
    if 60 <= zip_3 <= 69:
        return "CT"
    if 200 <= zip_3 <= 205:
        return "DC"
    if 197 <= zip_3 <= 199:
        return "DE"
    if 320 <= zip_3 <= 349:
        return "FL"
    if 300 <= zip_3 <= 319:
        return "GA"
    if 967 <= zip_3 <= 968:
        return "HI"
    if 500 <= zip_3 <= 528:
        return "IA"
    if 832 <= zip_3 <= 838:
        return "ID"
    if 600 <= zip_3 <= 629:
        return "IL"
    if 460 <= zip_3 <= 479:
        return "IN"
    if 660 <= zip_3 <= 679:
        return "KS"
    if 400 <= zip_3 <= 427:
        return "KY"
    if 700 <= zip_3 <= 714:
        return "LA"
    if 10 <= zip_3 <= 27:
        return "MA"
    if 210 <= zip_3 <= 219:
        return "MD"
    if 39 <= zip_3 <= 49:
        return "ME"
    if 480 <= zip_3 <= 499:
        return "MI"
    if 550 <= zip_3 <= 567:
        return "MN"
    if 630 <= zip_3 <= 658:
        return "MO"
    if 386 <= zip_3 <= 397:
        return "MS"
    if 590 <= zip_3 <= 599:
        return "MT"
    if 270 <= zip_3 <= 289:
        return "NC"
    if 580 <= zip_3 <= 588:
        return "ND"
    if 680 <= zip_3 <= 693:
        return "NE"
    if 30 <= zip_3 <= 38:
        return "NH"
    if 70 <= zip_3 <= 89:
        return "NJ"
    if 870 <= zip_3 <= 884:
        return "NM"
    if 889 <= zip_3 <= 898:
        return "NV"
    if 100 <= zip_3 <= 149:
        return "NY"
    if 430 <= zip_3 <= 459:
        return "OH"
    if 730 <= zip_3 <= 749:
        return "OK"
    if 970 <= zip_3 <= 979:
        return "OR"
    if 150 <= zip_3 <= 196:
        return "PA"
    if 28 <= zip_3 <= 29:
        return "RI"
    if 290 <= zip_3 <= 299:
        return "SC"
    if 570 <= zip_3 <= 577:
        return "SD"
    if 370 <= zip_3 <= 385:
        return "TN"
    if 750 <= zip_3 <= 799:
        return "TX"
    if 840 <= zip_3 <= 847:
        return "UT"
    if 220 <= zip_3 <= 246:
        return "VA"
    if 50 <= zip_3 <= 59:
        return "VT"
    if 980 <= zip_3 <= 994:
        return "WA"
    if 530 <= zip_3 <= 549:
        return "WI"
    if 247 <= zip_3 <= 268:
        return "WV"
    if 820 <= zip_3 <= 831:
        return "WY"

    return "UNKNOWN"


def load_and_filter_zips_by_state(file_path, target_states=None):
    """
    Loads all ZIP codes and segments them by the requested target states.
    target_states can be a string (e.g. 'CA') or a list (e.g. ['NY', 'NJ']).
    """
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please provide your source ZIP codes.")
        return []

    raw_zips = []

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, dtype=str)
            zip_col = [
                col
                for col in df.columns
                if "zip" in col.lower() or "postal" in col.lower()
            ]
            raw_zips = (
                df[zip_col].dropna().iloc[:, 0].tolist()
                if zip_col
                else df.iloc[:, 0].dropna().tolist()
            )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_zips = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading source file: {e}")
        return []

    # Apply geographic filtering rules
    if not target_states:
        return [str(z).zfill(5) for z in raw_zips]

    if isinstance(target_states, str):
        target_states = [target_states.upper()]
    else:
        target_states = [s.upper() for s in target_states]

    filtered_zips = []
    for z in raw_zips:
        padded_zip = str(z).zfill(5)
        if get_state_from_zip(padded_zip) in target_states:
            filtered_zips.append(padded_zip)

    return filtered_zips


def _extract_phone(text):
    """Finds a US phone number inside free-form SERP detail strings."""
    match = re.search(r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", text or "")
    return match.group(0) if match else "N/A"


def _extract_city_state(text):
    """Pulls 'City, ST' out of detail lines like 'Newberry, SCOpen' / 'Chicago, ILOpen 24 hours'."""
    if not text:
        return ""
    match = re.search(r"([A-Za-z][A-Za-z .'-]*?), ([A-Z]{2})", text)
    return f"{match.group(1).strip()}, {match.group(2)}" if match else ""


def _local_pack_to_lead(pack, zip_code):
    """Converts a SERP local_pack entry to a lead dict. Website is unknown from SERPs."""
    details = pack.get("details") or []
    joined = " | ".join(str(d) for d in details)
    category = re.sub(r"\d+ ?\+? ?years?.*$", "", str(details[1] if len(details) > 1 else ""))
    return {
        "Name": pack.get("title"),
        "Category": category or "N/A",
        "Phone": _extract_phone(joined),
        "Address": _extract_city_state(str(details[2] if len(details) > 2 else joined)),
        "Rating": pack.get("rating"),
        "Reviews Count": pack.get("rating_vote_count"),
        "ZIP Code": zip_code,
        "Website": "",  # local_packs from the SERP endpoint expose no website link
    }


def _organic_to_lead(result, zip_code):
    """Converts a SERP organic result to a lead dict (businesses that DO have websites)."""
    return {
        "Name": result.get("title"),
        "Category": "N/A",
        "Phone": "N/A",
        "Address": "",
        "Rating": None,
        "Reviews Count": None,
        "ZIP Code": zip_code,
        "Website": result.get("displayed_link") or result.get("link") or "",
    }


def _maps_result_to_lead(result, zip_code):
    """Converts a structured Google Maps result to a lead dict (has website field)."""
    return {
        "Name": result.get("title"),
        "Category": result.get("category", "N/A"),
        "Phone": result.get("phone", "N/A"),
        "Address": result.get("adress") or result.get("address") or "",
        "Rating": result.get("rating"),
        "Reviews Count": result.get("reviews"),
        "ZIP Code": zip_code,
        "Website": result.get("website", ""),
    }


def _scrape_maps_search(zip_code, business_type, min_rating, min_reviews):
    """
    Scrapes the structured Google Maps endpoint for one ZIP.

    Returns a list of lead dicts on success (even if empty). Raises
    MapsUnavailable when the endpoint is blocked or fails outright, so the
    caller can fall back to SERPs. On paid plans where Maps is accessible,
    this becomes the primary data source automatically.
    """
    payload = {
        "api_key": SCRAPERAPI_API_KEY,
        "query": f"{business_type} {zip_code}",
        "country_code": "us",
        "hl": "en",
        "limit": 100,
    }

    try:
        response = requests.get(SCRAPERAPI_MAPS_ENDPOINT, params=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(
            f" [Info] Maps endpoint unavailable for ZIP {zip_code} "
            f"(falling back to SERP for the rest of the run). Reason: {e}"
        )
        raise MapsUnavailable

    if isinstance(data, dict) and data.get("error"):
        print(
            f" [Info] Maps endpoint blocked for ZIP {zip_code} "
            "(falling back to SERP for the rest of the run)."
        )
        raise MapsUnavailable

    extracted_leads = []
    for result in data.get("map_results") or []:
        try:
            if result.get("rating") is not None and float(result["rating"]) < min_rating:
                continue
        except (TypeError, ValueError):
            pass
        try:
            if result.get("reviews") is not None and int(result["reviews"]) < min_reviews:
                continue
        except (TypeError, ValueError):
            pass
        extracted_leads.append(_maps_result_to_lead(result, zip_code))

    return extracted_leads


def scrape_batch(zip_batch, business_type, min_rating, min_reviews):
    """Scrapes each ZIP via structured Google Maps, falling back to Search SERPs."""
    global _MAPS_UNAVAILABLE
    extracted_leads = []

    for zip_code in zip_batch:
        maps_leads = None

        if not _MAPS_UNAVAILABLE:
            try:
                maps_leads = _scrape_maps_search(
                    zip_code, business_type, min_rating, min_reviews
                )
            except MapsUnavailable:
                _MAPS_UNAVAILABLE = True
                maps_leads = None

        if maps_leads is not None:
            extracted_leads.extend(maps_leads)
            continue

        payload = {
            "api_key": SCRAPERAPI_API_KEY,
            "query": f"{business_type} {zip_code}",
            "country_code": "us",
            "gl": "us",
            "hl": "en",
        }

        try:
            response = requests.get(SCRAPERAPI_SERP_ENDPOINT, params=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f" [Warning] ScraperAPI SERP request failed for ZIP {zip_code}: {e}")
            continue

        if isinstance(data, dict) and data.get("error"):
            print(f" [Warning] ScraperAPI SERP returned an error for ZIP {zip_code}: {data['error']}")
            continue

        for pack in data.get("local_packs") or []:
            try:
                if pack.get("rating") is not None and float(pack["rating"]) < min_rating:
                    continue
            except (TypeError, ValueError):
                pass
            try:
                if pack.get("rating_vote_count") is not None and int(pack["rating_vote_count"]) < min_reviews:
                    continue
            except (TypeError, ValueError):
                pass
            extracted_leads.append(_local_pack_to_lead(pack, zip_code))

        for result in data.get("organic_results") or []:
            extracted_leads.append(_organic_to_lead(result, zip_code))

    return extracted_leads


def append_to_file(new_leads, output_path):
    """Appends records iteratively to prevent data loss and drop duplicates."""
    if not new_leads:
        return

    new_df = pd.DataFrame(new_leads)

    if output_path.endswith(".xlsx"):
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_excel(output_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(
                    subset=["Name", "Phone"], keep="first", inplace=True
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
    # --- Workflow Configurations ---
    ZIP_INPUT_FILE = "us_zip_codes.txt"  # Master input file
    BUSINESS_KEYWORD = "Roofing Contractor"

    # GEOGRAPHIC FILTER SETTINGS
    # Change this to whichever state you want to process right now (e.g., "TX", "NY", "FL")
    # You can also pass a list like ["NY", "NJ", "CT"]
    TARGET_STATES = "TX"

    # Custom output file per state makes keeping track of operations effortless
    LEADS_OUTPUT_FILE = (
        f"leads_{TARGET_STATES.lower() if isinstance(TARGET_STATES, str) else 'multi_state'}.xlsx"
    )

    if SCRAPERAPI_API_KEY == "YOUR_SCRAPERAPI_API_KEY":
        print("[Abort]: Please configure your SCRAPERAPI_API_KEY before running.")
        return

    # Load and isolate only the ZIP codes that reside within our target boundaries
    state_zips = load_and_filter_zips_by_state(
        ZIP_INPUT_FILE, target_states=TARGET_STATES
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

    # Commencing batch looping execution
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

        print(f" -> Extracted {len(batch_results)} leads from SERP results.")
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
