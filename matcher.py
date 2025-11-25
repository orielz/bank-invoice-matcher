import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple, Set, Dict, List

import pandas as pd

TOL = 1.0          # amount tolerance (NIS)
VAT_RATE = 0.18    # 18% VAT
WINDOW_DAYS = 62   # ~2 months around invoice date
WINDOW_DAYS_CHEQUE = 120
MAX_CAND = 80      # limit number of bank candidates per invoice
WITHHOLDING_PERCENT_TOL = 0.1  # 0.1% tolerance for withholding rate detection

DATE_FORMATS = ("%d/%m/%Y", "%m/%d/%Y")
STATE_FILE = "withholding_state.json"
IGNORE_FILE = "ignore_descriptions.json"

# Default ignore patterns (used if no state file exists)
DEFAULT_IGNORE_PATTERNS = [
    "ביט' לאומי חיוב",
    "מס הכנסה",
    "מע\"מ חיוב",
]


def load_ignore_patterns(state_dir: Path) -> List[str]:
    """Load ignore patterns from state file."""
    ignore_path = state_dir / IGNORE_FILE
    if not ignore_path.exists():
        return DEFAULT_IGNORE_PATTERNS
    try:
        with open(ignore_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ignore_patterns", DEFAULT_IGNORE_PATTERNS)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_IGNORE_PATTERNS


def load_withholding_state(state_path: Path) -> Dict[str, float]:
    """Load withholding tax rates from state file."""
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("withholding_rates", {})
    except (json.JSONDecodeError, IOError):
        return {}


def save_withholding_state(state_path: Path, rates: Dict[str, float]) -> None:
    """Save withholding tax rates to state file."""
    data = {"withholding_rates": rates}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_date(s: str) -> Optional[datetime]:
    """Try several date formats; return None if parsing fails."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def invoice_supplier_key(row: pd.Series) -> Optional[str]:
    """Extract a 'supplier key' from invoice data."""
    det = str(row.get("Details", "")).strip()
    vend = str(row.get("Vendor/Credit Acc.", "")).strip()

    name = None
    if det and det != "-" and det.lower() != "nan":
        name = det
    elif vend and vend.lower() != "nan":
        if "(" in vend:
            name = vend.split("(", 1)[0].strip()
        else:
            name = vend
    else:
        return None

    parts = name.split()
    for p in parts:
        if len(p) >= 3:
            return p
    return parts[0] if parts else None


def exclude_bank_row(desc: str, amount: float, ignore_patterns: List[str] = None) -> bool:
    """Return True if a bank row should be excluded from matching."""
    if not isinstance(desc, str):
        return False

    # Use provided patterns or default
    patterns = ignore_patterns if ignore_patterns is not None else DEFAULT_IGNORE_PATTERNS

    for pattern in patterns:
        if pattern in desc:
            return True

    if pd.notna(amount) and abs(amount) < 50:
        return True
    return False


@dataclass
class Paths:
    invoices: Path
    bank: Path
    out_dir: Path


def load_data(paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    invoices = pd.read_csv(paths.invoices)
    bank = pd.read_csv(paths.bank, header=7)
    return invoices, bank


def invoice_vendor_key(row: pd.Series) -> Optional[str]:
    """Extract a 'vendor key' from Vendor/Credit Acc. field only."""
    vend = str(row.get("Vendor/Credit Acc.", "")).strip()
    if not vend or vend.lower() == "nan":
        return None

    if "(" in vend:
        name = vend.split("(", 1)[0].strip()
    else:
        name = vend

    parts = name.split()
    for p in parts:
        if len(p) >= 3:
            return p
    return parts[0] if parts else None


# Words to strip when extracting base vendor name
VENDOR_SUFFIX_WORDS = {
    "בע\"מ", "בעמ", "בע״מ",  # Ltd variations
    "חברה", "חברת",  # Company
    "ישראל",  # Israel
    "השקעות",  # Investments
    "שיווק", "והפצה",  # Marketing and distribution
    "מסחר",  # Commerce
    "עבודות",  # Works
    "שרותי", "שירותי",  # Services
    "תעשיות",  # Industries
    "מפעלי",  # Factories
    "אחים",  # Brothers
}


def extract_vendor_base_name(vendor: str) -> Optional[str]:
    """
    Extract the core business name from a vendor string.

    Examples:
    - "גאלה קלאסיק חדש (7818)" -> "גאלה"
    - "גאלה קלאסיק חומרי בניין בע\"מ (7124)" -> "גאלה"
    - "הנסון ישראל בע\"מ (7239)" -> "הנסון"
    - "אורן תומא מפעלי מתכת ומנופים בע\"מ (7129)" -> "אורן תומא"
    """
    if not vendor or pd.isna(vendor) or str(vendor).lower() == "nan":
        return None

    vendor = str(vendor).strip()
    if not vendor or vendor == "-":
        return None

    # Remove vendor code in parentheses at the end
    if "(" in vendor:
        vendor = vendor.split("(")[0].strip()

    # Split into words
    words = vendor.split()
    if not words:
        return None

    # Filter out suffix words and keep meaningful name parts
    core_words = []
    for word in words:
        # Skip common suffix words
        if word in VENDOR_SUFFIX_WORDS:
            continue
        # Skip single character words
        if len(word) < 2:
            continue
        core_words.append(word)

    if not core_words:
        # Fallback: return first word with length >= 2
        for w in words:
            if len(w) >= 2:
                return w
        return words[0] if words else None

    # Return first 1-2 significant words as the base name
    # This handles cases like "אורן תומא" (person name) vs "גאלה" (single word company)
    if len(core_words) >= 2:
        # Check if second word looks like part of a name (short word, no common suffix)
        second_word = core_words[1]
        # If second word is also short and not a common descriptor, include it
        if len(second_word) <= 5 and second_word not in {"קלאסיק", "חומרי", "בניין", "חדש", "טולס"}:
            return f"{core_words[0]} {core_words[1]}"

    return core_words[0]


def preprocess_invoices(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    invoices["Amount_clean"] = pd.to_numeric(
        invoices["Amount"].astype(str).str.replace(",", ""), errors="coerce"
    )
    invoices["Vat_clean"] = pd.to_numeric(
        invoices["Vat"].astype(str).str.replace(",", ""), errors="coerce"
    )
    invoices["inv_date"] = invoices["Transaction Date"].apply(parse_date)
    invoices["supplier_key"] = invoices.apply(invoice_supplier_key, axis=1)
    invoices["vendor_key"] = invoices.apply(invoice_vendor_key, axis=1)
    invoices["vendor_base_name"] = invoices["Vendor/Credit Acc."].apply(extract_vendor_base_name)
    invoices["matched"] = False
    invoices["match_type"] = None
    invoices["match_confidence"] = None  # high, medium, low
    invoices["bank_rows"] = None
    return invoices


def preprocess_bank(bank: pd.DataFrame, ignore_patterns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bank = bank.copy()
    bank["amount"] = pd.to_numeric(
        bank["Unnamed: 3"].astype(str).str.replace(",", ""), errors="coerce"
    )
    bank["date"] = bank["N/A"].apply(parse_date)
    bank["exclude"] = bank.apply(
        lambda r: exclude_bank_row(r["Unnamed: 2"], r["amount"], ignore_patterns), axis=1
    )

    bank_debits = bank[(bank["amount"] < 0) & (~bank["exclude"])].copy()
    bank_debits["mag"] = bank_debits["amount"].abs()
    bank_debits["key"] = bank_debits.apply(
        lambda r: (str(r["N/A"]), str(r["Unnamed: 2"]), float(r["amount"]) if not pd.isna(r["amount"]) else None),
        axis=1,
    )

    bank_match = bank_debits.copy()
    bank_match["matched"] = False
    return bank_debits, bank_match


def pass_single_amount_match(invoices: pd.DataFrame, bank_match: pd.DataFrame) -> None:
    """Pass 1: invoice -> single bank row by amount, filtered by supplier_key.

    Confidence levels:
    - high: supplier_key matches bank description
    - medium: vendor_key matches bank description (fallback)

    No amount-only matching - supplier name is required.
    """
    for inv_idx, inv in invoices[invoices["Amount_clean"].notna()].iterrows():
        if inv["matched"]:
            continue
        amount = inv["Amount_clean"]
        supplier_key = inv["supplier_key"]
        vendor_key = inv["vendor_key"]

        all_cand = bank_match[~bank_match["matched"]]

        # Try supplier_key first (high confidence)
        matched_row = None
        confidence = None

        if supplier_key:
            cand = all_cand[all_cand["Unnamed: 2"].astype(str).str.contains(supplier_key, na=False)]
            exact = cand[(cand["mag"] - amount).abs() <= TOL]
            if not exact.empty:
                exact = exact.copy()
                if inv["inv_date"] is not None:
                    exact["date_diff"] = (exact["date"] - inv["inv_date"]).abs().dt.days
                    exact = exact.sort_values(["date_diff", "date"])
                matched_row = exact.index[0]
                confidence = "high"

        # Try vendor_key if supplier_key didn't match (medium confidence)
        if matched_row is None and vendor_key and vendor_key != supplier_key:
            cand = all_cand[all_cand["Unnamed: 2"].astype(str).str.contains(vendor_key, na=False)]
            exact = cand[(cand["mag"] - amount).abs() <= TOL]
            if not exact.empty:
                exact = exact.copy()
                if inv["inv_date"] is not None:
                    exact["date_diff"] = (exact["date"] - inv["inv_date"]).abs().dt.days
                    exact = exact.sort_values(["date_diff", "date"])
                matched_row = exact.index[0]
                confidence = "medium"

        if matched_row is not None:
            bank_match.at[matched_row, "matched"] = True
            invoices.at[inv_idx, "matched"] = True
            invoices.at[inv_idx, "match_type"] = "single"
            invoices.at[inv_idx, "match_confidence"] = confidence
            invoices.at[inv_idx, "bank_rows"] = [int(matched_row)]


def pass_combo_two_bank(invoices: pd.DataFrame, bank_match: pd.DataFrame) -> None:
    """Pass 2: invoice -> 2 bank rows (same supplier, within WINDOW_DAYS)."""
    for inv_idx, inv in invoices[(~invoices["matched"]) & invoices["Amount_clean"].notna()].iterrows():
        amount = inv["Amount_clean"]
        key = inv["supplier_key"]
        inv_date = inv["inv_date"]

        cand = bank_match[~bank_match["matched"]].copy()
        if inv_date is not None:
            cand = cand[cand["date"].notna() & (cand["date"].sub(inv_date).abs().dt.days <= WINDOW_DAYS)]
        if key:
            cand = cand[cand["Unnamed: 2"].astype(str).str.contains(key, na=False)]
        if len(cand) < 2:
            continue
        if len(cand) > MAX_CAND:
            cand = cand.sort_values("mag").head(MAX_CAND)

        idxs = list(cand.index)
        mags = cand["mag"].to_dict()

        found = False
        for i, j in combinations(idxs, 2):
            s = mags[i] + mags[j]
            if abs(s - amount) <= TOL:
                bank_match.at[i, "matched"] = True
                bank_match.at[j, "matched"] = True
                invoices.at[inv_idx, "matched"] = True
                invoices.at[inv_idx, "match_type"] = "combo2_sum"
                invoices.at[inv_idx, "bank_rows"] = [int(i), int(j)]
                found = True
                break
        if found:
            continue

        for i, j in combinations(idxs, 2):
            a, b = mags[i], mags[j]
            bigger, smaller = (a, b) if a >= b else (b, a)
            if smaller == 0:
                continue
            ratio = smaller / bigger
            if 0.175 <= ratio <= 0.185 and abs(a + b - amount) <= TOL:
                bank_match.at[i, "matched"] = True
                bank_match.at[j, "matched"] = True
                invoices.at[inv_idx, "matched"] = True
                invoices.at[inv_idx, "match_type"] = "combo2_sum_vatratio"
                invoices.at[inv_idx, "bank_rows"] = [int(i), int(j)]
                found = True
                break


def pass_single_vat_factor(invoices: pd.DataFrame, bank_match: pd.DataFrame) -> None:
    """Pass 3: invoice -> single bank row by VAT factor: abs(bank)*1.18 ~= invoice."""
    for inv_idx, inv in invoices[(~invoices["matched"]) & invoices["Amount_clean"].notna()].iterrows():
        amount = inv["Amount_clean"]
        key = inv["supplier_key"]
        inv_date = inv["inv_date"]

        cand = bank_match[~bank_match["matched"]].copy()
        if inv_date is not None:
            cand = cand[cand["date"].notna() & (cand["date"].sub(inv_date).abs().dt.days <= WINDOW_DAYS)]
        if key:
            cand = cand[cand["Unnamed: 2"].astype(str).str.contains(key, na=False)]
        if cand.empty:
            continue

        factor = 1.0 + VAT_RATE
        diff = (cand["mag"] * factor - amount).abs()
        best = diff.idxmin()
        if diff.loc[best] <= TOL:
            bank_match.at[best, "matched"] = True
            invoices.at[inv_idx, "matched"] = True
            invoices.at[inv_idx, "match_type"] = "single_vat_factor"
            invoices.at[inv_idx, "bank_rows"] = [int(best)]


def pass_multi_invoice_to_bank(invoices: pd.DataFrame, bank_match: pd.DataFrame, max_invoices: int = 6) -> None:
    """Match multiple invoices (2-6) from the same supplier to a single bank payment."""
    # Group unmatched invoices by vendor
    unmatched_inv = invoices[(~invoices["matched"]) & invoices["Amount_clean"].notna()].copy()

    if unmatched_inv.empty:
        return

    # Group by vendor
    vendor_groups = unmatched_inv.groupby("Vendor/Credit Acc.")

    for vendor, vendor_invoices in vendor_groups:
        if len(vendor_invoices) < 2:
            continue

        # Get supplier/vendor keys for this vendor
        sample_row = vendor_invoices.iloc[0]
        supplier_key = sample_row["supplier_key"]
        vendor_key = sample_row["vendor_key"]

        # Find unmatched bank rows matching this supplier
        bank_cand = bank_match[~bank_match["matched"]].copy()

        matched_bank = pd.DataFrame()
        if supplier_key:
            matched_bank = bank_cand[bank_cand["Unnamed: 2"].astype(str).str.contains(supplier_key, na=False)]
        if matched_bank.empty and vendor_key:
            matched_bank = bank_cand[bank_cand["Unnamed: 2"].astype(str).str.contains(vendor_key, na=False)]

        if matched_bank.empty:
            continue

        # Limit to reasonable number of invoices to avoid combinatorial explosion
        # Sort by amount descending to prioritize larger invoices
        vendor_invoices = vendor_invoices.sort_values("Amount_clean", ascending=False).head(15)

        inv_idxs = list(vendor_invoices.index)
        inv_amounts = vendor_invoices["Amount_clean"].to_dict()

        # Build a set of target bank amounts for quick lookup
        bank_targets = {bank_idx: bank_row["mag"] for bank_idx, bank_row in matched_bank.iterrows()}

        # Try combinations of 2 to max_invoices
        for num_inv in range(2, min(max_invoices + 1, len(inv_idxs) + 1)):
            found_match = False
            for inv_combo in combinations(inv_idxs, num_inv):
                combo_sum = sum(inv_amounts[i] for i in inv_combo)

                # Check if any bank row matches this sum
                for bank_idx, target in bank_targets.items():
                    if bank_match.at[bank_idx, "matched"]:
                        continue

                    if abs(target - combo_sum) <= TOL:
                        # Found a match!
                        bank_match.at[bank_idx, "matched"] = True

                        for inv_idx in inv_combo:
                            invoices.at[inv_idx, "matched"] = True
                            invoices.at[inv_idx, "match_type"] = f"multi_{num_inv}_to_1"
                            invoices.at[inv_idx, "match_confidence"] = "high"
                            invoices.at[inv_idx, "bank_rows"] = [int(bank_idx)]

                        found_match = True
                        break

                if found_match:
                    break

            if found_match:
                break  # Move to next vendor after finding a match


def pass_multi_invoice_cross_vendor(invoices: pd.DataFrame, bank_match: pd.DataFrame, max_invoices: int = 6) -> None:
    """
    Match multiple invoices from DIFFERENT vendors (but same base name) to a single bank payment.

    This handles cases where a company has multiple vendor codes (e.g., גאלה קלאסיק חדש and
    גאלה קלאסיק חומרי בניין) but payments are made to a single account.
    """
    unmatched_inv = invoices[(~invoices["matched"]) & invoices["Amount_clean"].notna()].copy()

    if unmatched_inv.empty:
        return

    # Group by base vendor name (not exact vendor)
    unmatched_inv = unmatched_inv[unmatched_inv["vendor_base_name"].notna()]
    if unmatched_inv.empty:
        return

    base_name_groups = unmatched_inv.groupby("vendor_base_name")

    for base_name, base_invoices in base_name_groups:
        # Only process if we have invoices from multiple different vendor codes
        unique_vendors = base_invoices["Vendor/Credit Acc."].nunique()
        if unique_vendors < 2:
            # Already handled by pass_multi_invoice_to_bank
            continue

        if len(base_invoices) < 2:
            continue

        # Get the base name for bank description matching
        search_key = base_name.split()[0] if base_name else None  # Use first word for search
        if not search_key or len(search_key) < 2:
            continue

        # Find unmatched bank rows matching this base name
        bank_cand = bank_match[~bank_match["matched"]].copy()
        matched_bank = bank_cand[bank_cand["Unnamed: 2"].astype(str).str.contains(search_key, na=False)]

        if matched_bank.empty:
            continue

        # Limit invoices to avoid combinatorial explosion
        # For cross-vendor, allow more invoices since this is a secondary pass
        # and we need to capture small invoices that might complete a sum
        base_invoices = base_invoices.sort_values("Amount_clean", ascending=False).head(20)

        inv_idxs = list(base_invoices.index)
        inv_amounts = base_invoices["Amount_clean"].to_dict()
        inv_vendors = base_invoices["Vendor/Credit Acc."].to_dict()

        # Build bank targets
        bank_targets = {bank_idx: bank_row["mag"] for bank_idx, bank_row in matched_bank.iterrows()}

        # Try combinations of 2 to max_invoices
        for num_inv in range(2, min(max_invoices + 1, len(inv_idxs) + 1)):
            found_match = False
            for inv_combo in combinations(inv_idxs, num_inv):
                # Require at least 2 different vendors in the combination (otherwise pass_multi_invoice_to_bank handles it)
                vendors_in_combo = set(inv_vendors[i] for i in inv_combo)
                if len(vendors_in_combo) < 2:
                    continue

                combo_sum = sum(inv_amounts[i] for i in inv_combo)

                # Check if any bank row matches this sum
                for bank_idx, target in bank_targets.items():
                    if bank_match.at[bank_idx, "matched"]:
                        continue

                    if abs(target - combo_sum) <= TOL:
                        # Found a match!
                        bank_match.at[bank_idx, "matched"] = True

                        for inv_idx in inv_combo:
                            invoices.at[inv_idx, "matched"] = True
                            invoices.at[inv_idx, "match_type"] = f"multi_{num_inv}_cross_vendor"
                            invoices.at[inv_idx, "match_confidence"] = "medium"  # Medium because cross-vendor
                            invoices.at[inv_idx, "bank_rows"] = [int(bank_idx)]

                        found_match = True
                        break

                if found_match:
                    break

            if found_match:
                break  # Move to next base name after finding a match


def build_invoice_bank_matches(invoices: pd.DataFrame, bank_match: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Group multi-invoice matches by bank_row to show related invoices together
    multi_invoice_groups: Dict[int, List[int]] = {}
    for inv_idx, inv in invoices[invoices["matched"]].iterrows():
        match_type = inv.get("match_type", "")
        if match_type and match_type.startswith("multi_"):
            for r in inv["bank_rows"]:
                if r not in multi_invoice_groups:
                    multi_invoice_groups[r] = []
                multi_invoice_groups[r].append(inv_idx)

    for inv_idx, inv in invoices[invoices["matched"]].iterrows():
        for r in inv["bank_rows"]:
            br = bank_match.loc[r]
            match_type = inv.get("match_type", "")

            # For multi-invoice matches, show all related invoice indices
            related_invoices = ""
            if match_type and match_type.startswith("multi_") and r in multi_invoice_groups:
                related_invoices = ", ".join(str(i) for i in multi_invoice_groups[r])

            rows.append(
                {
                    "Invoice_Index": inv_idx,
                    "Match_Type": match_type,
                    "Confidence": inv.get("match_confidence", "high"),
                    "Related_Invoices": related_invoices,
                    "Invoice_Date": inv.get("Transaction Date"),
                    "Invoice_Details": inv.get("Details"),
                    "Invoice_Vendor": inv.get("Vendor/Credit Acc."),
                    "Invoice_Amount": inv["Amount_clean"],
                    "Bank_Row_Index": r,
                    "Bank_Date": br["N/A"],
                    "Bank_Description": br["Unnamed: 2"],
                    "Bank_Amount": br["amount"],
                }
            )
    return pd.DataFrame(rows)


def pass_cheque_multi_invoices(invoices: pd.DataFrame, bank_debits: pd.DataFrame, max_invoices: int = 6) -> pd.DataFrame:
    """Handle cheques: bank rows whose description contains 'משיכת שיק'.

    Matches cheques to 1-6 invoices from the same vendor.
    """
    cheques = bank_debits[bank_debits["Unnamed: 2"].astype(str).str.contains("משיכת שיק", na=False)].copy()

    matches = []
    matched_cheques = set()

    for bidx, brow in cheques.iterrows():
        if bidx in matched_cheques:
            continue

        target = abs(brow["amount"])
        cheque_date = parse_date(brow["N/A"])

        cand = invoices[invoices["Amount_clean"].notna()].copy()
        if cheque_date is not None:
            cand = cand[
                cand["inv_date"].notna()
                & (cand["inv_date"] <= cheque_date)
                & ((cheque_date - cand["inv_date"]).dt.days <= WINDOW_DAYS_CHEQUE)
            ]
        cand = cand[cand["Amount_clean"] <= target + TOL]
        if cand.empty:
            continue

        # Group candidates by vendor
        vendor_groups = cand.groupby("Vendor/Credit Acc.")

        found_match = False
        for vendor, vendor_cand in vendor_groups:
            if vendor_cand.empty:
                continue

            # Limit candidates per vendor
            vendor_cand = vendor_cand.sort_values("Amount_clean", ascending=False).head(15)

            idxs = list(vendor_cand.index)
            amounts = vendor_cand["Amount_clean"].to_dict()

            # Try combinations of 1 to max_invoices (including single invoice matches)
            for num_inv in range(1, min(max_invoices + 1, len(idxs) + 1)):
                for inv_combo in combinations(idxs, num_inv):
                    combo_sum = sum(amounts[i] for i in inv_combo)

                    if abs(combo_sum - target) <= TOL:
                        # Found a match!
                        matched_cheques.add(bidx)

                        match_data = {
                            "Cheque_Bank_Index": bidx,
                            "Cheque_Date": brow["N/A"],
                            "Cheque_Description": brow["Unnamed: 2"],
                            "Cheque_Amount": brow["amount"],
                            "Num_Invoices": num_inv,
                            "Invoice_Indices": list(inv_combo),
                            "Invoice_Vendor": vendor,
                            "Invoice_Amounts": [amounts[i] for i in inv_combo],
                            "Invoice_Total": combo_sum,
                        }
                        matches.append(match_data)
                        found_match = True
                        break

                if found_match:
                    break

            if found_match:
                break

    if not matches:
        return pd.DataFrame()

    return pd.DataFrame(matches)


def get_supplier_full_name(row: pd.Series) -> str:
    """Get full supplier name from invoice for display purposes."""
    vend = str(row.get("Vendor/Credit Acc.", "")).strip()
    if vend and vend.lower() != "nan" and vend != "-":
        return vend
    det = str(row.get("Details", "")).strip()
    if det and det.lower() != "nan" and det != "-":
        return det
    return "Unknown"


def detect_withholding_candidates(
    invoices: pd.DataFrame,
    bank_match: pd.DataFrame,
) -> Dict[str, List[Dict]]:
    """
    Detect suppliers with multiple unmatched invoices that may have withholding tax.

    Returns a dict mapping supplier name to list of potential matches with computed rates.
    """
    candidates: Dict[str, List[Dict]] = {}

    # Get unmatched invoices with amounts
    unmatched_inv = invoices[(~invoices["matched"]) & invoices["Amount_clean"].notna()]

    for inv_idx, inv in unmatched_inv.iterrows():
        amount = inv["Amount_clean"]
        supplier_key = inv["supplier_key"]
        vendor_key = inv["vendor_key"]
        inv_date = inv["inv_date"]
        supplier_name = get_supplier_full_name(inv)

        # Find potential bank matches (same supplier, different amount)
        all_cand = bank_match[~bank_match["matched"]]

        # Try to find bank rows with matching supplier
        bank_cand = pd.DataFrame()
        used_key = None

        if supplier_key:
            bank_cand = all_cand[all_cand["Unnamed: 2"].astype(str).str.contains(supplier_key, na=False)]
            used_key = supplier_key

        if bank_cand.empty and vendor_key and vendor_key != supplier_key:
            bank_cand = all_cand[all_cand["Unnamed: 2"].astype(str).str.contains(vendor_key, na=False)]
            used_key = vendor_key

        if bank_cand.empty:
            continue

        # Apply date window filter
        if inv_date is not None:
            bank_cand = bank_cand[
                bank_cand["date"].notna() &
                (bank_cand["date"].sub(inv_date).abs().dt.days <= WINDOW_DAYS)
            ]

        if bank_cand.empty:
            continue

        # Look for bank rows where amount is less than invoice (withholding reduces payment)
        for bank_idx, bank_row in bank_cand.iterrows():
            bank_amount = bank_row["mag"]

            # Withholding means bank pays less than invoice
            if bank_amount >= amount:
                continue

            # Calculate the withholding percentage
            # If invoice is 1000 and bank paid 950, withholding rate is 5%
            withholding_rate = (amount - bank_amount) / amount

            # Only consider reasonable withholding rates (0.5% to 30%)
            if not (0.005 <= withholding_rate <= 0.30):
                continue

            if supplier_name not in candidates:
                candidates[supplier_name] = []

            candidates[supplier_name].append({
                "inv_idx": inv_idx,
                "inv_date": inv.get("Transaction Date"),
                "inv_amount": amount,
                "bank_idx": bank_idx,
                "bank_date": bank_row["N/A"],
                "bank_description": bank_row["Unnamed: 2"],
                "bank_amount": bank_amount,
                "withholding_rate": withholding_rate,
                "supplier_key": used_key,
            })

    return candidates


def analyze_withholding_patterns(
    candidates: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    """
    Analyze candidates to find suppliers with consistent withholding rates.

    Returns suppliers that have multiple mismatches with the same percentage gap.
    """
    consistent_suppliers: Dict[str, Dict] = {}

    for supplier_name, matches in candidates.items():
        if len(matches) < 2:
            continue

        # Group by similar withholding rates
        rates = [m["withholding_rate"] for m in matches]

        # Check if all rates are consistent (within WITHHOLDING_PERCENT_TOL)
        avg_rate = sum(rates) / len(rates)

        all_consistent = all(
            abs(r - avg_rate) * 100 <= WITHHOLDING_PERCENT_TOL
            for r in rates
        )

        if all_consistent:
            consistent_suppliers[supplier_name] = {
                "avg_rate": avg_rate,
                "matches": matches,
                "count": len(matches),
            }

    return consistent_suppliers


def interactive_withholding_confirmation(
    consistent_suppliers: Dict[str, Dict],
    existing_rates: Dict[str, float],
) -> Dict[str, float]:
    """
    Interactively ask user to confirm withholding tax rates for each supplier.

    Returns updated withholding rates dict.
    """
    new_rates = existing_rates.copy()

    for supplier_name, data in consistent_suppliers.items():
        avg_rate = data["avg_rate"]
        matches = data["matches"]
        rate_percent = avg_rate * 100

        # Skip if we already have this supplier with the same rate
        if supplier_name in existing_rates:
            existing_percent = existing_rates[supplier_name] * 100
            if abs(existing_percent - rate_percent) < WITHHOLDING_PERCENT_TOL:
                print(f"\n[Supplier already configured: {supplier_name} @ {existing_percent:.1f}%]")
                continue

        print("\n" + "=" * 70)
        print(f"Potential withholding tax detected for: {supplier_name}")
        print("=" * 70)
        print(f"\nFound {len(matches)} mismatches with consistent ~{rate_percent:.2f}% difference:\n")

        for i, m in enumerate(matches, 1):
            print(f"  {i}. Invoice: {m['inv_amount']:,.2f} NIS ({m['inv_date']})")
            print(f"     Bank:    {m['bank_amount']:,.2f} NIS ({m['bank_date']})")
            print(f"     Gap:     {m['inv_amount'] - m['bank_amount']:,.2f} NIS ({m['withholding_rate']*100:.2f}%)")
            print()

        print(f"This likely indicates withholding tax (ניכוי במקור) of ~{rate_percent:.2f}%")
        print()

        while True:
            response = input(f"Treat {supplier_name} as having {rate_percent:.2f}% withholding? [y/n/custom]: ").strip().lower()

            if response == 'y':
                new_rates[supplier_name] = avg_rate
                print(f"✓ Saved: {supplier_name} @ {rate_percent:.2f}%")
                break
            elif response == 'n':
                print(f"✗ Skipped: {supplier_name}")
                break
            elif response == 'custom':
                try:
                    custom_rate = float(input("Enter custom withholding rate (e.g., 5 for 5%): "))
                    new_rates[supplier_name] = custom_rate / 100
                    print(f"✓ Saved: {supplier_name} @ {custom_rate:.2f}%")
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                print("Please enter 'y', 'n', or 'custom'")

    return new_rates


def pass_withholding_match(
    invoices: pd.DataFrame,
    bank_match: pd.DataFrame,
    withholding_rates: Dict[str, float],
) -> None:
    """
    Match invoices to bank rows using known withholding tax rates.

    For each unmatched invoice, check if:
    expected_paid = invoice_amount * (1 - withholding_rate)
    matches a bank debit within tolerance.
    """
    if not withholding_rates:
        return

    for inv_idx, inv in invoices[(~invoices["matched"]) & invoices["Amount_clean"].notna()].iterrows():
        amount = inv["Amount_clean"]
        supplier_key = inv["supplier_key"]
        vendor_key = inv["vendor_key"]
        inv_date = inv["inv_date"]
        supplier_name = get_supplier_full_name(inv)

        # Check if this supplier has a known withholding rate
        withholding_rate = None
        for known_supplier, rate in withholding_rates.items():
            # Match by supplier name (partial match)
            if known_supplier in supplier_name or supplier_name in known_supplier:
                withholding_rate = rate
                break

        if withholding_rate is None:
            continue

        # Calculate expected payment after withholding
        expected_paid = amount * (1 - withholding_rate)

        all_cand = bank_match[~bank_match["matched"]]

        # Try supplier_key first
        matched_row = None
        confidence = None

        if supplier_key:
            cand = all_cand[all_cand["Unnamed: 2"].astype(str).str.contains(supplier_key, na=False)]
            if inv_date is not None:
                cand = cand[cand["date"].notna() & (cand["date"].sub(inv_date).abs().dt.days <= WINDOW_DAYS)]

            exact = cand[(cand["mag"] - expected_paid).abs() <= TOL]
            if not exact.empty:
                exact = exact.copy()
                if inv_date is not None:
                    exact["date_diff"] = (exact["date"] - inv_date).abs().dt.days
                    exact = exact.sort_values(["date_diff", "date"])
                matched_row = exact.index[0]
                confidence = "high"

        # Try vendor_key if supplier_key didn't match
        if matched_row is None and vendor_key and vendor_key != supplier_key:
            cand = all_cand[all_cand["Unnamed: 2"].astype(str).str.contains(vendor_key, na=False)]
            if inv_date is not None:
                cand = cand[cand["date"].notna() & (cand["date"].sub(inv_date).abs().dt.days <= WINDOW_DAYS)]

            exact = cand[(cand["mag"] - expected_paid).abs() <= TOL]
            if not exact.empty:
                exact = exact.copy()
                if inv_date is not None:
                    exact["date_diff"] = (exact["date"] - inv_date).abs().dt.days
                    exact = exact.sort_values(["date_diff", "date"])
                matched_row = exact.index[0]
                confidence = "medium"

        if matched_row is not None:
            bank_match.at[matched_row, "matched"] = True
            invoices.at[inv_idx, "matched"] = True
            invoices.at[inv_idx, "match_type"] = f"withholding_{withholding_rate*100:.1f}%"
            invoices.at[inv_idx, "match_confidence"] = confidence
            invoices.at[inv_idx, "bank_rows"] = [int(matched_row)]


def build_unmatched_invoices(invoices: pd.DataFrame, cheque_best: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame of invoices that were not matched to any bank transaction."""
    # Get invoice indices matched via cheques
    cheque_matched_indices: Set[int] = set()
    if not cheque_best.empty and "Invoice_Indices" in cheque_best.columns:
        for _, row in cheque_best.iterrows():
            indices_str = str(row["Invoice_Indices"])
            # Parse the list string like "[1, 82]"
            try:
                import ast
                indices = ast.literal_eval(indices_str)
                cheque_matched_indices.update(indices)
            except:
                pass

    # Unmatched = not matched via regular passes AND not matched via cheques
    unmatched_mask = (~invoices["matched"]) & (~invoices.index.isin(cheque_matched_indices))
    unmatched_invoices = invoices[unmatched_mask].copy()

    # Select and rename columns for cleaner output
    output_cols = []
    col_mapping = {
        "Transaction Date": "Invoice_Date",
        "Details": "Details",
        "Vendor/Credit Acc.": "Vendor",
        "Amount_clean": "Amount",
        "Vat_clean": "VAT",
    }

    result_rows = []
    for idx, row in unmatched_invoices.iterrows():
        result_rows.append({
            "Invoice_Index": idx,
            "Invoice_Date": row.get("Transaction Date", ""),
            "Details": row.get("Details", ""),
            "Vendor": row.get("Vendor/Credit Acc.", ""),
            "Amount": row.get("Amount_clean", ""),
            "VAT": row.get("Vat_clean", ""),
        })

    return pd.DataFrame(result_rows)


def build_unmatched(
    bank_debits: pd.DataFrame,
    invoice_bank_matches: pd.DataFrame,
    cheque_best: pd.DataFrame,
) -> pd.DataFrame:
    def key_from_invoice_bank(row) -> Tuple[str, str, float]:
        return (str(row["Bank_Date"]), str(row["Bank_Description"]), float(row["Bank_Amount"]))

    matched_from_invoice: Set[Tuple[str, str, float]] = set()
    if not invoice_bank_matches.empty:
        matched_from_invoice = {key_from_invoice_bank(r) for _, r in invoice_bank_matches.iterrows()}

    def key_from_cheque(row) -> Tuple[str, str, float]:
        return (str(row["Cheque_Date"]), str(row["Cheque_Description"]), float(row["Cheque_Amount"]))

    matched_from_cheques: Set[Tuple[str, str, float]] = set()
    if not cheque_best.empty and "Cheque_Date" in cheque_best.columns:
        matched_from_cheques = {key_from_cheque(r) for _, r in cheque_best.iterrows()}

    all_matched_keys = matched_from_invoice | matched_from_cheques

    unmatched_clean = bank_debits[~bank_debits["key"].isin(all_matched_keys)].copy()
    return unmatched_clean


def analyze_unmatched_bank_transactions(
    unmatched_bank: pd.DataFrame,
    invoices: pd.DataFrame,
    invoice_bank_matches: pd.DataFrame,
    cheque_best: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Analyze top unmatched bank transactions by amount and provide detailed conclusions.

    For each unmatched transaction, checks:
    1. If there are invoice combinations that could match (including already-matched invoices)
    2. If an invoice was "used twice" (matched elsewhere but could also match here)
    3. If there's a close match with small difference
    4. Possible reasons for no match
    """
    import ast

    # Get all matched invoice indices
    matched_indices = set(invoice_bank_matches["Invoice_Index"].tolist())
    cheque_matched_indices = set()
    cheque_matches_detail = {}  # Map invoice idx to cheque info

    for _, row in cheque_best.iterrows():
        try:
            indices = ast.literal_eval(str(row["Invoice_Indices"]))
            for idx in indices:
                cheque_matched_indices.add(idx)
                cheque_matches_detail[idx] = {
                    "cheque_date": row["Cheque_Date"],
                    "cheque_desc": row["Cheque_Description"],
                    "cheque_amount": row["Cheque_Amount"],
                    "all_indices": indices,
                }
        except:
            pass

    all_matched = matched_indices | cheque_matched_indices

    # Get invoice match details for regular matches
    invoice_match_detail = {}
    for _, row in invoice_bank_matches.iterrows():
        inv_idx = row["Invoice_Index"]
        invoice_match_detail[inv_idx] = {
            "bank_date": row["Bank_Date"],
            "bank_desc": row["Bank_Description"],
            "bank_amount": row["Bank_Amount"],
            "match_type": row["Match_Type"],
        }

    # Sort unmatched by absolute amount and take top N
    unmatched_sorted = unmatched_bank.copy()
    unmatched_sorted["abs_amount"] = unmatched_sorted["amount"].abs()
    unmatched_sorted = unmatched_sorted.sort_values("abs_amount", ascending=False).head(top_n)

    # Prepare invoice data for searching
    invoices_with_amount = invoices[invoices["Amount_clean"].notna()].copy()

    results = []
    total = len(unmatched_sorted)
    import time
    start_time = time.time()

    for i, (bank_idx, bank_row) in enumerate(unmatched_sorted.iterrows(), 1):
        elapsed = time.time() - start_time
        avg_time = elapsed / i if i > 1 else 0
        remaining = avg_time * (total - i)
        print(f"  Analyzing {i}/{total}: Amount {bank_row['amount']:.2f} | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s", end="\r")
        target = abs(bank_row["amount"])
        bank_date = bank_row.get("N/A", "")
        bank_desc = str(bank_row.get("Unnamed: 2", ""))

        analysis = {
            "Bank_Index": bank_idx,
            "Date": bank_date,
            "Description": bank_desc,
            "Amount": bank_row["amount"],
            "Analysis": "",
            "Potential_Match": "",
            "Issue_Type": "",
        }

        # Extract potential vendor name from bank description
        vendor_keywords = []
        for word in bank_desc.split():
            if len(word) >= 3 and word not in {"הע.", "הע", "לבע", "בסניף", "העברה", "חיוב", "משיכת", "שיק"}:
                vendor_keywords.append(word)

        # Search for matching invoices
        found_combinations = []
        used_twice_combos = []
        close_matches = []

        # Try to find invoice combinations (1-6 invoices)
        for vendor_key in vendor_keywords[:3]:  # Limit search to first 3 keywords
            matching_invoices = invoices_with_amount[
                invoices_with_amount["Vendor/Credit Acc."].str.contains(vendor_key, na=False) |
                invoices_with_amount["Details"].str.contains(vendor_key, na=False)
            ]

            if matching_invoices.empty:
                continue

            # Limit to 20 invoices to avoid combinatorial explosion
            matching_invoices = matching_invoices.sort_values("Amount_clean", ascending=False).head(20)
            inv_list = [(idx, row["Amount_clean"]) for idx, row in matching_invoices.iterrows()]

            # Try combinations (limit to 4 invoices for speed)
            for num_inv in range(1, min(5, len(inv_list) + 1)):
                for combo in combinations(inv_list, num_inv):
                    combo_sum = sum(x[1] for x in combo)
                    diff = abs(combo_sum - target)

                    if diff <= TOL:
                        combo_indices = [x[0] for x in combo]
                        # Check if any invoice in combo is already matched
                        matched_in_combo = [idx for idx in combo_indices if idx in all_matched]

                        if matched_in_combo:
                            used_twice_combos.append({
                                "indices": combo_indices,
                                "amounts": [x[1] for x in combo],
                                "sum": combo_sum,
                                "matched_elsewhere": matched_in_combo,
                            })
                        else:
                            found_combinations.append({
                                "indices": combo_indices,
                                "amounts": [x[1] for x in combo],
                                "sum": combo_sum,
                            })
                    elif diff <= 100:  # Close match
                        combo_indices = [x[0] for x in combo]
                        close_matches.append({
                            "indices": combo_indices,
                            "amounts": [x[1] for x in combo],
                            "sum": combo_sum,
                            "diff": diff,
                        })

        # Build analysis text
        analysis_parts = []
        potential_match = ""
        issue_type = ""

        if found_combinations:
            # There's an available combination - why wasn't it matched?
            combo = found_combinations[0]
            potential_match = f"Invoices {combo['indices']} = {combo['sum']}"
            issue_type = "AVAILABLE_COMBO_NOT_MATCHED"
            analysis_parts.append(f"Found available invoice combination: {combo['indices']} summing to {combo['sum']}")
            analysis_parts.append("Possible reasons: vendor name mismatch in bank description, or date window issue")

        elif used_twice_combos:
            # Invoice(s) already used elsewhere
            combo = used_twice_combos[0]
            potential_match = f"Invoices {combo['indices']} = {combo['sum']}"
            issue_type = "INVOICE_USED_TWICE"

            analysis_parts.append(f"POTENTIAL DOUBLE PAYMENT or MISSING INVOICE")
            analysis_parts.append(f"Combination found: {combo['indices']} = {' + '.join(str(a) for a in combo['amounts'])} = {combo['sum']}")
            analysis_parts.append("")

            # Detail where each matched invoice was used
            for idx in combo['matched_elsewhere']:
                if idx in cheque_matches_detail:
                    detail = cheque_matches_detail[idx]
                    analysis_parts.append(f"Invoice [{idx}] already used in cheque match:")
                    analysis_parts.append(f"  Cheque: {detail['cheque_date']} | {detail['cheque_desc']} | {detail['cheque_amount']}")
                    analysis_parts.append(f"  With invoices: {detail['all_indices']}")
                elif idx in invoice_match_detail:
                    detail = invoice_match_detail[idx]
                    analysis_parts.append(f"Invoice [{idx}] already used in bank match:")
                    analysis_parts.append(f"  Bank: {detail['bank_date']} | {detail['bank_desc']} | {detail['bank_amount']}")
                    analysis_parts.append(f"  Match type: {detail['match_type']}")

            analysis_parts.append("")
            analysis_parts.append("Conclusions:")
            analysis_parts.append("1. Data issue - duplicate invoice may be missing from data")
            analysis_parts.append("2. Double payment - same invoice paid twice by mistake")
            analysis_parts.append("3. Different vendor - payment may be for different vendor with similar name")

        elif close_matches:
            # Close but not exact match
            close = min(close_matches, key=lambda x: x['diff'])
            potential_match = f"Invoices {close['indices']} = {close['sum']} (diff: {close['diff']:.2f})"
            issue_type = "CLOSE_MATCH"
            analysis_parts.append(f"Close match found (within {close['diff']:.2f} NIS):")
            analysis_parts.append(f"Invoices {close['indices']} = {close['sum']}")
            analysis_parts.append("Possible reasons: rounding, fees, or partial payment")

        else:
            # No match found at all
            issue_type = "NO_MATCH_FOUND"
            if vendor_keywords:
                analysis_parts.append(f"Searched for vendors: {', '.join(vendor_keywords[:3])}")
            analysis_parts.append("No matching invoice combination found")
            analysis_parts.append("Possible reasons:")
            analysis_parts.append("1. Invoice not in system (missing data)")
            analysis_parts.append("2. Payment for non-invoice item (salary, tax, etc.)")
            analysis_parts.append("3. Vendor name in bank doesn't match invoice vendor")

        analysis["Analysis"] = "\n".join(analysis_parts)
        analysis["Potential_Match"] = potential_match
        analysis["Issue_Type"] = issue_type

        results.append(analysis)

    print()  # New line after progress
    total_time = time.time() - start_time
    print(f"  Analysis complete in {total_time:.1f}s")

    return pd.DataFrame(results)


@dataclass
class ConfigPaths:
    invoices: Path
    bank: Path
    out_dir: Path
    interactive: bool = False
    state_file: Path = None


def generate_html(df: pd.DataFrame, title: str, output_path: Path, color_by_confidence: bool = False) -> None:
    """Generate an HTML file with a styled table from a DataFrame."""
    html_template = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            background-color: #e8f5e9;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: inline-block;
        }}
        .legend {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: inline-block;
            margin-left: 20px;
            padding: 5px 10px;
            border-radius: 3px;
        }}
        .legend-high {{ background-color: #c8e6c9; }}
        .legend-medium {{ background-color: #fff9c4; }}
        .legend-low {{ background-color: #ffcdd2; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 8px;
            text-align: right;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ddd;
            text-align: right;
        }}
        tr:hover {{
            filter: brightness(0.95);
        }}
        tr.confidence-high {{
            background-color: #c8e6c9;
        }}
        tr.confidence-medium {{
            background-color: #fff9c4;
        }}
        tr.confidence-low {{
            background-color: #ffcdd2;
        }}
        .amount {{
            font-family: monospace;
            direction: ltr;
            text-align: left;
        }}
        .negative {{
            color: #c62828;
        }}
        .positive {{
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">Total rows: {row_count}</div>
    {legend}
    {table}
</body>
</html>"""

    legend_html = ""
    if color_by_confidence and not df.empty and "Confidence" in df.columns:
        legend_html = """
    <div class="legend">
        <strong>Confidence:</strong>
        <span class="legend-item legend-high">High - Supplier match</span>
        <span class="legend-item legend-medium">Medium - Vendor match</span>
        <span class="legend-item legend-low">Low - Amount only</span>
    </div>"""

    if df.empty:
        table_html = "<p>No data</p>"
    else:
        if color_by_confidence and "Confidence" in df.columns:
            # Build HTML table manually with row classes
            table_html = '<table class="data-table">\n<thead>\n<tr>'
            for col in df.columns:
                table_html += f"<th>{col}</th>"
            table_html += "</tr>\n</thead>\n<tbody>\n"
            for _, row in df.iterrows():
                conf = row.get("Confidence", "high")
                if conf is None:
                    conf = "high"
                table_html += f'<tr class="confidence-{conf}">'
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        val = ""
                    table_html += f"<td>{val}</td>"
                table_html += "</tr>\n"
            table_html += "</tbody>\n</table>"
        else:
            table_html = df.to_html(index=False, classes="data-table", escape=False)

    html_content = html_template.format(
        title=title,
        row_count=len(df),
        legend=legend_html,
        table=table_html,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_analysis_html(df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Generate an HTML file with styled analysis table, with special formatting for multiline analysis."""
    html_template = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #e91e63;
            padding-bottom: 10px;
        }}
        .stats {{
            background-color: #fce4ec;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: inline-block;
        }}
        .legend {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: inline-block;
            margin-left: 20px;
            padding: 5px 10px;
            border-radius: 3px;
        }}
        .issue-invoice-used-twice {{ background-color: #ffcdd2; }}
        .issue-available-combo {{ background-color: #fff9c4; }}
        .issue-close-match {{ background-color: #c8e6c9; }}
        .issue-no-match {{ background-color: #e0e0e0; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #e91e63;
            color: white;
            padding: 12px 8px;
            text-align: right;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ddd;
            text-align: right;
            vertical-align: top;
        }}
        tr:hover {{
            filter: brightness(0.95);
        }}
        tr.issue-INVOICE_USED_TWICE {{
            background-color: #ffcdd2;
        }}
        tr.issue-AVAILABLE_COMBO_NOT_MATCHED {{
            background-color: #fff9c4;
        }}
        tr.issue-CLOSE_MATCH {{
            background-color: #c8e6c9;
        }}
        tr.issue-NO_MATCH_FOUND {{
            background-color: #e0e0e0;
        }}
        .analysis-cell {{
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            text-align: right;
            direction: ltr;
            max-width: 600px;
        }}
        .amount {{
            font-family: monospace;
            direction: ltr;
            text-align: left;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">Total analyzed: {row_count}</div>
    <div class="legend">
        <strong>Issue Types:</strong>
        <span class="legend-item issue-invoice-used-twice">Invoice Used Twice - Potential double payment</span>
        <span class="legend-item issue-available-combo">Available Combo - Should have matched</span>
        <span class="legend-item issue-close-match">Close Match - Small difference</span>
        <span class="legend-item issue-no-match">No Match Found</span>
    </div>
    {table}
</body>
</html>"""

    if df.empty:
        table_html = "<p>No data</p>"
    else:
        table_html = '<table class="data-table">\n<thead>\n<tr>'
        for col in df.columns:
            table_html += f"<th>{col}</th>"
        table_html += "</tr>\n</thead>\n<tbody>\n"

        for _, row in df.iterrows():
            issue_type = row.get("Issue_Type", "")
            table_html += f'<tr class="issue-{issue_type}">'
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    val = ""
                if col == "Analysis":
                    # Escape HTML and preserve newlines
                    val = str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    table_html += f'<td class="analysis-cell">{val}</td>'
                elif col == "Amount":
                    table_html += f'<td class="amount">{val}</td>'
                else:
                    table_html += f"<td>{val}</td>"
            table_html += "</tr>\n"
        table_html += "</tbody>\n</table>"

    html_content = html_template.format(
        title=title,
        row_count=len(df),
        table=table_html,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def run(paths: ConfigPaths) -> None:
    invoices_raw, bank_raw = load_data(Paths(paths.invoices, paths.bank, paths.out_dir))
    invoices = preprocess_invoices(invoices_raw)

    # Load ignore patterns from state directory
    state_dir = paths.state_file.parent if paths.state_file else Path("state")
    ignore_patterns = load_ignore_patterns(state_dir)
    print(f"Loaded {len(ignore_patterns)} ignore pattern(s) from {state_dir / IGNORE_FILE}")
    for pattern in ignore_patterns:
        print(f"  - {pattern}")

    bank_debits, bank_match = preprocess_bank(bank_raw, ignore_patterns)

    # Load existing withholding rates
    state_path = paths.state_file or paths.out_dir / STATE_FILE
    withholding_rates = load_withholding_state(state_path)

    if withholding_rates:
        print(f"Loaded {len(withholding_rates)} withholding rate(s) from {state_path}")
        for supplier, rate in withholding_rates.items():
            print(f"  - {supplier}: {rate*100:.1f}%")

    # Standard matching passes
    pass_single_amount_match(invoices, bank_match)
    pass_combo_two_bank(invoices, bank_match)
    pass_single_vat_factor(invoices, bank_match)

    # Multi-invoice to single bank payment (2-6 invoices)
    pass_multi_invoice_to_bank(invoices, bank_match)

    # Cross-vendor multi-invoice matching (same base name, different vendor codes)
    pass_multi_invoice_cross_vendor(invoices, bank_match)

    # Apply known withholding rates (before interactive detection)
    if withholding_rates:
        pass_withholding_match(invoices, bank_match, withholding_rates)

    # Interactive withholding detection
    if paths.interactive:
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE: Analyzing unmatched items for withholding tax patterns...")
        print("=" * 70)

        candidates = detect_withholding_candidates(invoices, bank_match)
        consistent_suppliers = analyze_withholding_patterns(candidates)

        if consistent_suppliers:
            print(f"\nFound {len(consistent_suppliers)} supplier(s) with consistent withholding patterns.")
            withholding_rates = interactive_withholding_confirmation(
                consistent_suppliers, withholding_rates
            )

            # Save updated rates
            save_withholding_state(state_path, withholding_rates)
            print(f"\nWithholding rates saved to {state_path}")

            # Re-run withholding match with new rates
            pass_withholding_match(invoices, bank_match, withholding_rates)
        else:
            print("\nNo consistent withholding patterns detected.")

    invoice_bank_matches = build_invoice_bank_matches(invoices, bank_match)
    cheque_best = pass_cheque_multi_invoices(invoices, bank_debits)
    unmatched_clean = build_unmatched(bank_debits, invoice_bank_matches, cheque_best)
    unmatched_invoices = build_unmatched_invoices(invoices, cheque_best)

    # Analyze top 100 unmatched bank transactions (by amount)
    print("Analyzing top 100 unmatched bank transactions...")
    unmatched_analysis = analyze_unmatched_bank_transactions(
        unmatched_clean, invoices, invoice_bank_matches, cheque_best, top_n=100
    )

    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV files
    invoice_bank_matches.to_csv(out_dir / "final_invoice_bank_matches.csv", index=False)
    cheque_best.to_csv(out_dir / "final_cheque_best_matches.csv", index=False)
    unmatched_clean.to_csv(out_dir / "unmatched_clean_final.csv", index=False)
    unmatched_invoices.to_csv(out_dir / "unmatched_invoices.csv", index=False)
    unmatched_analysis.to_csv(out_dir / "unmatched_analysis.csv", index=False)

    # Generate HTML files
    generate_html(
        invoice_bank_matches,
        "Invoice-Bank Matches",
        out_dir / "final_invoice_bank_matches.html",
        color_by_confidence=True,
    )
    generate_html(
        cheque_best,
        "Cheque Best Matches",
        out_dir / "final_cheque_best_matches.html",
    )
    # Select cleaner columns for unmatched display
    unmatched_display = unmatched_clean[["N/A", "Unnamed: 2", "amount"]].copy()
    unmatched_display.columns = ["Date", "Description", "Amount"]
    generate_html(
        unmatched_display,
        "Unmatched Bank Debits",
        out_dir / "unmatched_clean_final.html",
    )
    generate_html(
        unmatched_invoices,
        "Unmatched Invoices",
        out_dir / "unmatched_invoices.html",
    )

    # Generate analysis HTML with special formatting
    generate_analysis_html(
        unmatched_analysis,
        "Unmatched Bank Analysis - Top 100 by Amount",
        out_dir / "unmatched_analysis.html",
    )

    print("Wrote:")
    print(f"  - {out_dir / 'final_invoice_bank_matches.csv'}")
    print(f"  - {out_dir / 'final_cheque_best_matches.csv'}")
    print(f"  - {out_dir / 'unmatched_clean_final.csv'}")
    print(f"  - {out_dir / 'unmatched_invoices.csv'}")
    print(f"  - {out_dir / 'unmatched_analysis.csv'}")
    print(f"  - {out_dir / 'final_invoice_bank_matches.html'}")
    print(f"  - {out_dir / 'final_cheque_best_matches.html'}")
    print(f"  - {out_dir / 'unmatched_clean_final.html'}")
    print(f"  - {out_dir / 'unmatched_invoices.html'}")
    print(f"  - {out_dir / 'unmatched_analysis.html'}")


def parse_args() -> ConfigPaths:
    p = argparse.ArgumentParser(description="Bank–Invoice matcher")
    p.add_argument("--invoices", type=str, default="data/invoices.csv", help="Path to invoices CSV")
    p.add_argument("--bank", type=str, default="data/bank.csv", help="Path to bank CSV")
    p.add_argument("--out-dir", type=str, default="output", help="Output directory for result CSVs")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode to detect and configure withholding tax rates"
    )
    p.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to withholding state file (default: <out-dir>/withholding_state.json)"
    )
    args = p.parse_args()

    return ConfigPaths(
        invoices=Path(args.invoices),
        bank=Path(args.bank),
        out_dir=Path(args.out_dir),
        interactive=args.interactive,
        state_file=Path(args.state_file) if args.state_file else None,
    )


if __name__ == "__main__":
    paths = parse_args()
    run(paths)
