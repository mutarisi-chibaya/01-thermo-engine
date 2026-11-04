#!/usr/bin/env bash
# Download the raw NIST sources used by build_reference_data.py into data/raw/.
# Run from 01-thermo-engine:  bash scripts/fetch_sources.sh
# Files that already exist are kept, so the saved sources stay fixed;
# set FORCE=1 to re-download everything.
set -euo pipefail
UA="pyrothermo-data-fetch"
mkdir -p data/raw/nist data/raw/janaf

get() {  # get <output file> <url>
  if [[ -s "$1" && -z "${FORCE:-}" ]]; then return; fi
  curl -fsS -A "$UA" -o "$1" "$2"
}

# key  WebBook ID  Mask (1 = gas-phase thermochemistry, 2 = condensed-phase)
while read -r key id mask; do
  get "data/raw/nist/$key.html" "https://webbook.nist.gov/cgi/cbook.cgi?ID=$id&Units=SI&Mask=$mask"
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
Al C7429905 2
Al2O3 C1344281 2
Cr C7440473 2
Cr2O3 C1308389 2
Mn C7439965 2
Ni C7440020 2
Si C7440213 2
SiO2 C14808607 2
SiO2_liquid C7631869 2
LIST
# SiO2: the quartz page (CAS 14808-60-7) has the solid fits; the generic SiO2 page
# (CAS 7631-86-9) has only the liquid fit.
# MnO and NiO are absent: neither the WebBook nor NIST-JANAF has data for them.

for t in Fe-003 Fe-020 Fe-032 Fe-030 C-002 C-093 C-095 H-050 H-064 O-029 \
         Al-001 Al-101 Cr-001 Cr-014 Mn-001 Ni-001 Si-001 O-037; do
  get "data/raw/janaf/$t.txt" "https://janaf.nist.gov/tables/$t.txt"
done
echo "done"
