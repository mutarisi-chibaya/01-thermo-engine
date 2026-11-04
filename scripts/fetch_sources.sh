#!/usr/bin/env bash
# Download the raw NIST sources used by build_reference_data.py into data/raw/.
# Run from 01-thermo-engine:  bash scripts/fetch_sources.sh
set -euo pipefail
UA="pyrothermo-data-fetch"
mkdir -p data/raw/nist data/raw/janaf
while read -r key id mask; do
  curl -fsS -A "$UA" -o "data/raw/nist/$key.html" \
    "https://webbook.nist.gov/cgi/cbook.cgi?ID=$id&Units=SI&Mask=$mask"
done <<'LIST'
Fe C7439896 2
FeO C1345251 2
Fe3O4 C1309382 2
Fe2O3 C1317608 2
C C7440440 2
CO C630080 1
CO2 C124389 1
H2 C1333740 1
H2O C7732185 1
O2 C7782447 1
LIST
for t in Fe-003 Fe-020 Fe-032 Fe-030 C-002 C-093 C-095 H-050 H-064 O-029; do
  curl -fsS -A "$UA" -o "data/raw/janaf/$t.txt" "https://janaf.nist.gov/tables/$t.txt"
done
echo "done"
