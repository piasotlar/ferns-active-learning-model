import pandas as pd
import re

possible_segments = ['10', 'M5', 'RNA3a', 'RNA12', 'S', 'dsRNA3', '6', 'Seg5B', 'RNA6', '5', 'RNA5b', 'B', 
                     'RNA6a', '9', 'RNA2', 'dsRNA8', 'S11', 'Seg5A', 'dsRNA7', '4', 'RNA3', 'RNA9', '11', 
                     'DNA-S.1', 'DNA-B', 'RNA6b', 'L', 'Sa', '2', 'DNA-S.2', '8', 'S7', 'S8', 'S10', 'partial', 
                     '7', 'L3', 'L2', 'Mb', 'SegB', 'Seg9', 'dsRNA6', 'L1', 'RNA4', 'Seg1', 'Seg3', 'M2', 'RNA6c', 
                     'RNA8', 'DNA-U2', 'DNA-A', 'M1', 'DNA-N', 'RNA5a', 'M3', 'dsRNA5', 'DNA-S', 'M6', '1', 
                     'Seg10', 'DNA-U4', 'Seg12', 'S9', 'Seg1-3', 'S5', 'S4', 'dsRNA2', 'DNA-U1', 'S2', 'DNA-C', 'Seg7', 
                     'RNA3b', 'S6', 'M4', 'Seg4', 'dsRNA1', 'DNA-M', 'RNA5', 'M', 'DNA-U3', 'RNA11', 'Seg4-1', 'S1', 'Ma', 
                     'Seg6', 'DNA-R', 'Seg5', 'RNA7', 'Seg2', 'Seg8', 'SegA', 'Seg11', 'Seg4-2', 'Sb', 'dsRNA4', 'S12', 
                     'RNA10', 'A', 'DNA-gamma', '3', 'S3', 'RNA1']


def find_host(host):
    if pd.isna(host):
        return "Unknown"

    host = str(host).lower()

    if "plants" in host:
        return "Yes"
    return "No"

def find_length(length_table, family, genus=None, segment=None):
    if segment and genus:
        match = length_table[
            (length_table['Genus'] == genus) &
            (length_table['Segment'] == segment)
        ]
    else:
        if genus:
            match = length_table[length_table['Genus'] == genus]
        else:
            match = length_table[length_table['Family'] == family]

    if not match.empty:
        row = match.iloc[0]
        _min = row['min_length_rounded']
        _max = row['max_length_rounded']
        host = row['Host source']
        return _min, _max, host
    else:
        return None, None, None


def evaluate_taxonomy(row, family_col, genus_col):
    family = row[family_col]
    genus = row[genus_col]

    if pd.isna(family):
        return "Unknown", "Unknown"

    if pd.notna(genus):
        min_len, max_len, host = find_length(
            genuses,
            family,
            genus
        )
    else:
        min_len, max_len, host = find_length(familys, family)

    if pd.notna(min_len) and pd.notna(max_len) and pd.notna(row["LENGTH"]):
        if min_len < row["LENGTH"] < max_len:
            length_consistent = "Yes"
        else:
            length_consistent = "No"
    else:
        length_consistent = "Unknown"

    plant_host = find_host(host)
    return length_consistent, plant_host

table = pd.read_csv('table.tsv', header=0, sep='\t')
genuses_segmented = pd.read_csv('genus_segmented_lengths_rounded_with_host.txt', header=0, sep='\t')
genuses = pd.read_csv('genus_lengths_rounded_with_host.txt', header=0, sep='\t')
familys = pd.read_csv('family_lengths_rounded_with_host.txt', header=0, sep='\t')

table["LENGTH"] = pd.to_numeric(table["LENGTH"], errors="coerce")

for df in [genuses_segmented, genuses, familys]:
    df["min_length_rounded"] = pd.to_numeric(df["min_length_rounded"], errors="coerce")
    df["max_length_rounded"] = pd.to_numeric(df["max_length_rounded"], errors="coerce")

table["CONSISTENT LENGTH (tax)"] = "Unknown"
table["PLANT HOST (tax)"] = "Unknown"

for idx, row in table.iterrows():
    tax_length, tax_host = evaluate_taxonomy(
        row,
        family_col="FAMILY",
        genus_col="GENUS"
    )

    table.loc[idx, "CONSISTENT LENGTH (tax)"] = tax_length
    table.loc[idx, "PLANT HOST (tax)"] = tax_host
   
table.to_csv("table_extended.tsv", sep="\t", index=False)