import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple, Set

import pandas as pd

TOL = 1.0          # amount tolerance (NIS)
VAT_RATE = 0.18    # 18% VAT
WINDOW_DAYS = 62   # ~2 months around invoice date
WINDOW_DAYS_CHEQUE = 120
MAX_CAND = 80      # limit number of bank candidates per invoice

DATE_FORMATS = ("%d/%m/%Y", "%m/%d/%Y")


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


def exclude_bank_row(desc: str, amount: float) -> bool:
    """Return True if a bank row should be excluded from matching."""
    if not isinstance(desc, str):
        return False
    if "ביט' לאומי חיוב" in desc:
        return True
    if "מס הכנסה" in desc:
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
    invoices["matched"] = False
    invoices["match_type"] = None
    invoices["bank_rows"] = None
    return invoices


def preprocess_bank(bank: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bank = bank.copy()
    bank["amount"] = pd.to_numeric(
        bank["Unnamed: 3"].astype(str).str.replace(",", ""), errors="coerce"
    )
    bank["date"] = bank["N/A"].apply(parse_date)
    bank["exclude"] = bank.apply(
        lambda r: exclude_bank_row(r["Unnamed: 2"], r["amount"]), axis=1
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
    """Pass 1: invoice -> single bank row by amount, filtered by supplier_key."""
    for inv_idx, inv in invoices[invoices["Amount_clean"].notna()].iterrows():
        if inv["matched"]:
            continue
        amount = inv["Amount_clean"]
        key = inv["supplier_key"]
        cand = bank_match[~bank_match["matched"]]
        if key:
            cand = cand[cand["Unnamed: 2"].astype(str).str.contains(key, na=False)]
        if cand.empty:
            continue
        exact = cand[(cand["mag"] - amount).abs() <= TOL]
        if exact.empty:
            continue

        exact = exact.copy()
        if inv["inv_date"] is not None:
            exact["date_diff"] = (exact["date"] - inv["inv_date"]).abs().dt.days
            exact = exact.sort_values(["date_diff", "date"])
        row = exact.index[0]

        bank_match.at[row, "matched"] = True
        invoices.at[inv_idx, "matched"] = True
        invoices.at[inv_idx, "match_type"] = "single"
        invoices.at[inv_idx, "bank_rows"] = [int(row)]


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


def build_invoice_bank_matches(invoices: pd.DataFrame, bank_match: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inv_idx, inv in invoices[invoices["matched"]].iterrows():
        for r in inv["bank_rows"]:
            br = bank_match.loc[r]
            rows.append(
                {
                    "Invoice_Index": inv_idx,
                    "Match_Type": inv["match_type"],
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


def pass_cheque_two_invoices(invoices: pd.DataFrame, bank_debits: pd.DataFrame) -> pd.DataFrame:
    """Handle cheques: bank rows whose description contains 'משיכת שיק'."""
    cheques = bank_debits[bank_debits["Unnamed: 2"].astype(str).str.contains("משיכת שיק", na=False)].copy()

    pairs = []
    for bidx, brow in cheques.iterrows():
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
        if len(cand) < 2:
            continue

        idxs = list(cand.index)
        amounts = cand["Amount_clean"].to_dict()
        vendors = cand["Vendor/Credit Acc."].to_dict()

        for i, j in combinations(idxs, 2):
            if str(vendors[i]) != str(vendors[j]):
                continue
            s = amounts[i] + amounts[j]
            if abs(s - target) <= TOL:
                inv1 = invoices.loc[i]
                inv2 = invoices.loc[j]
                pairs.append(
                    {
                        "Cheque_Bank_Index": bidx,
                        "Cheque_Date": brow["N/A"],
                        "Cheque_Description": brow["Unnamed: 2"],
                        "Cheque_Amount": brow["amount"],
                        "Inv1_Index": i,
                        "Inv1_Date": inv1["Transaction Date"],
                        "Inv1_Details": inv1["Details"],
                        "Inv1_Vendor": inv1["Vendor/Credit Acc."],
                        "Inv1_Amount": inv1["Amount_clean"],
                        "Inv2_Index": j,
                        "Inv2_Date": inv2["Transaction Date"],
                        "Inv2_Details": inv2["Details"],
                        "Inv2_Vendor": inv2["Vendor/Credit Acc."],
                        "Inv2_Amount": inv2["Amount_clean"],
                    }
                )

    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs)
    best_rows = []
    for cheque_idx, grp in df.groupby("Cheque_Bank_Index"):
        best_rows.append(grp.iloc[0])
    return pd.DataFrame(best_rows)


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
    if not cheque_best.empty:
        matched_from_cheques = {key_from_cheque(r) for _, r in cheque_best.iterrows()}

    all_matched_keys = matched_from_invoice | matched_from_cheques

    unmatched_clean = bank_debits[~bank_debits["key"].isin(all_matched_keys)].copy()
    return unmatched_clean


@dataclass
class ConfigPaths:
    invoices: Path
    bank: Path
    out_dir: Path


def run(paths: ConfigPaths) -> None:
    invoices_raw, bank_raw = load_data(Paths(paths.invoices, paths.bank, paths.out_dir))
    invoices = preprocess_invoices(invoices_raw)
    bank_debits, bank_match = preprocess_bank(bank_raw)

    pass_single_amount_match(invoices, bank_match)
    pass_combo_two_bank(invoices, bank_match)
    pass_single_vat_factor(invoices, bank_match)

    invoice_bank_matches = build_invoice_bank_matches(invoices, bank_match)
    cheque_best = pass_cheque_two_invoices(invoices, bank_debits)
    unmatched_clean = build_unmatched(bank_debits, invoice_bank_matches, cheque_best)

    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    invoice_bank_matches.to_csv(out_dir / "final_invoice_bank_matches.csv", index=False)
    cheque_best.to_csv(out_dir / "final_cheque_best_matches.csv", index=False)
    unmatched_clean.to_csv(out_dir / "unmatched_clean_final.csv", index=False)

    print("Wrote:")
    print(f"  - {out_dir / 'final_invoice_bank_matches.csv'}")
    print(f"  - {out_dir / 'final_cheque_best_matches.csv'}")
    print(f"  - {out_dir / 'unmatched_clean_final.csv'}")


def parse_args() -> ConfigPaths:
    p = argparse.ArgumentParser(description="Bank–Invoice matcher")
    p.add_argument("--invoices", type=str, default="data/invoices.csv", help="Path to invoices CSV")
    p.add_argument("--bank", type=str, default="data/bank.csv", help="Path to bank CSV")
    p.add_argument("--out-dir", type=str, default="output", help="Output directory for result CSVs")
    args = p.parse_args()

    return ConfigPaths(
        invoices=Path(args.invoices),
        bank=Path(args.bank),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    paths = parse_args()
    run(paths)
