# Bank–Invoice Matching Logic (Spec for Agents)

This file describes the logic expected from `matcher.py`.

## Pre-processing

### Invoices

For each row:

- Amount_clean: numeric from `Amount`
- Vat_clean: numeric from `Vat`
- inv_date: parsed from `Transaction Date` (DD/MM/YYYY or MM/DD/YYYY)
- supplier_key:
  - Prefer `Details`; if empty, use `Vendor/Credit Acc.`
  - Strip from first `(` onward
  - Split into words, take first word with length >= 3
  - If none, take first word if exists

### Bank

For each row:

- amount: numeric from `Unnamed: 3`
- date: parsed from `N/A`
- exclude row if:
  - description contains "ביט' לאומי חיוב"
  - description contains "מס הכנסה"
  - abs(amount) < 50
- Work only with debit rows: amount < 0 and not excluded
- For each debit row:
  - mag = abs(amount)
  - key = (N/A, Unnamed: 2, amount)

## Matching passes (order matters)

1. Invoice -> single bank row (`single`)
2. Invoice -> 2 bank rows (`combo2_sum`, `combo2_sum_vatratio`)
3. Invoice -> single bank row via VAT factor (`single_vat_factor`)
4. Cheque bank row -> 2 invoices (`final_cheque_best_matches`)

Bank rows marked as matched in earlier passes are not reused later.

## Pass 1: single (`single`)

For each invoice with Amount_clean and not yet matched:

- Candidates: bank rows where matched == False
- If supplier_key exists: keep only rows where description contains supplier_key
- Keep rows with abs(mag - Amount_clean) <= TOL (TOL = 1.0)
- If multiple, choose the one closest in date to inv_date
- Mark invoice as matched, type = single, bank_rows = [index]
- Mark bank row as matched

## Pass 2: 2 bank rows per invoice (`combo2_sum`, `combo2_sum_vatratio`)

For each unmatched invoice:

- Candidates: unmatched bank rows
  - If inv_date exists: abs(bank_date - inv_date).days <= WINDOW_DAYS (62)
  - If supplier_key exists: description contains supplier_key
- If fewer than 2 candidates: skip
- If many candidates: limit to MAX_CAND (e.g. 80) by smallest amounts

Check all pairs (i, j):

- Rule A: abs(mag[i] + mag[j] - Amount_clean) <= TOL
  - If found: match_type = combo2_sum
- Rule B (if no A found):
  - Let bigger, smaller be max/min of mag[i], mag[j]
  - ratio = smaller / bigger
  - If 0.175 <= ratio <= 0.185 AND abs(mag[i] + mag[j] - Amount_clean) <= TOL:
    - match_type = combo2_sum_vatratio

In both cases:

- Mark both bank rows as matched
- Mark invoice as matched, bank_rows = [i, j]

The same-supplier rule is enforced via supplier_key filter.

## Pass 3: VAT factor (`single_vat_factor`)

For each still-unmatched invoice:

- Candidates: unmatched bank rows, within WINDOW_DAYS of inv_date, and containing supplier_key (if any)
- For each candidate: diff = abs(mag * (1 + VAT_RATE) - Amount_clean), VAT_RATE = 0.18
- Pick candidate with minimum diff
- If diff <= TOL:
  - match_type = single_vat_factor
  - bank_rows = [index]
  - mark row as matched

## Pass 4: Cheques -> 2 invoices

Identify cheque rows: description contains "משיכת שיק".

For each cheque row:

- target = abs(amount)
- cheque_date = parsed date
- Candidate invoices:
  - Amount_clean not null
  - inv_date <= cheque_date
  - (cheque_date - inv_date).days <= WINDOW_DAYS_CHEQUE (120)
  - Amount_clean <= target + TOL

For all pairs of candidate invoices (i, j):

- Require same vendor: Vendor/Credit Acc. equal
- Require abs(Amount_i + Amount_j - target) <= TOL

All such pairs are valid; we keep the first per cheque as the best.

Write these to `final_cheque_best_matches.csv`.

## Unmatched

Build two sets of matched keys:

- From invoice->bank matches: (Bank_Date, Bank_Description, Bank_Amount)
- From cheque matches: (Cheque_Date, Cheque_Description, Cheque_Amount)

Then:

unmatched_clean_final.csv = all debit rows whose key is not in either set.

## Agent rules

- Do not change TOL, VAT_RATE, or WINDOW_DAYS without documenting.
- Never allow a multi-line match (2 bank rows or 2 invoices) to mix suppliers.
- Prefer no match over a wrong match.
- Add new Match_Type values if you create new logic.
