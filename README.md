# Bank–Invoice Matcher

This repository contains a small Python tool that automatically matches:

- Bank debit transactions (תנועות חובה בבנק)
- Supplier invoices (חשבוניות ספקים)

It implements the reconciliation logic we designed together:

- Exact amount 1:1 matches
- Two-bank-rows-to-one-invoice combinations (same supplier)
- VAT logic (net * 1.18 ~= invoice amount)
- Cheque -> two invoices of the same vendor
- Exclusion of certain bank rows (credit card, income tax, very small amounts)

## Usage

1. Put your CSV exports here:

   - `data/invoices.csv`  – invoice export
   - `data/bank.csv`      – bank movements export

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python matcher.py
```

Outputs will be written to the `output/` directory:

- `final_invoice_bank_matches.csv`
- `final_cheque_best_matches.csv`
- `unmatched_clean_final.csv`
