from pathlib import Path
import pandas as pd

# ============================================================
# NASTAVITVE
# ============================================================

INPUT_FILE = Path("cv_3.xlsx")
BATCH_DIR = Path("batches")
OUTPUT_FILE = Path("cv_3_completed.xlsx")

KEY_COLUMNS = ["SAMPLE", "CONTIG NAME"]
ITERATION_COLUMN = "ITERATION_ADDED"


# ============================================================
# FUNKCIJA: ali vrednost manjka?
# ============================================================

def is_missing(value):
    """
    True samo za dejansko manjkajoče vrednosti.
    0, False, "No", "Unknown", "/" itd. se NE štejejo kot prazne.
    """
    if pd.isna(value):
        return True

    if isinstance(value, str) and value.strip() == "":
        return True

    return False


# ============================================================
# PREBERI CV_3
# ============================================================

df = pd.read_excel(INPUT_FILE)

# Počisti imena stolpcev
df.columns = df.columns.astype(str).str.strip()

# Preveri obvezne stolpce
for col in KEY_COLUMNS + [ITERATION_COLUMN]:
    if col not in df.columns:
        raise ValueError(
            f"V {INPUT_FILE} manjka obvezen stolpec: {col}"
        )


print(f"Prebran: {INPUT_FILE}")
print(f"Število contigov: {len(df)}")
print(f"Začetno število stolpcev: {len(df.columns)}")


# ============================================================
# CACHE ZA BATCH DATOTEKE
# ============================================================

batch_cache = {}

matched_contigs = 0
not_found_contigs = 0
filled_cells = 0
added_columns = set()
missing_batch_files = set()


# ============================================================
# GREMO ČEZ VSAK CONTIG V CV_3
# ============================================================

for idx, row in df.iterrows():

    iteration = row[ITERATION_COLUMN]
    sample = row["SAMPLE"]
    contig = row["CONTIG NAME"]

    # ----------------------------------------
    # Če ITERATION_ADDED manjka
    # ----------------------------------------

    if pd.isna(iteration):
        print(
            f"SKIP: {sample} / {contig} "
            f"nima ITERATION_ADDED"
        )
        continue

    iteration = int(iteration)

    # ----------------------------------------
    # ITERATION_ADDED - 2
    # ----------------------------------------

    if iteration == 22:
        # posebna CV datoteka
        batch_file = BATCH_DIR / "to_label_iteration_CV.xlsx"

    elif iteration == 0:
        # originalna tabela
        batch_file = BATCH_DIR / "original_table.xlsx"

    else:
        # vse ostale iteracije: iteration - 2
        batch_iteration = iteration - 2
        batch_file = BATCH_DIR / f"to_label_iteration_{batch_iteration}.xlsx"

    # ----------------------------------------
    # Preveri obstoj datoteke
    # ----------------------------------------

    if not batch_file.exists():

        if batch_file not in missing_batch_files:
            print(f"MANJKA DATOTEKA: {batch_file}")
            missing_batch_files.add(batch_file)

        continue

    # ----------------------------------------
    # Preberi batch samo enkrat
    # ----------------------------------------

    if batch_file not in batch_cache:

        print(f"Berem: {batch_file}")

        batch_df = pd.read_excel(batch_file)

        batch_df.columns = (
            batch_df.columns
            .astype(str)
            .str.strip()
        )

        # preveri ključe
        for col in KEY_COLUMNS:
            if col not in batch_df.columns:
                raise ValueError(
                    f"{batch_file} nima stolpca {col}"
                )

        # index = SAMPLE + CONTIG NAME
        batch_df = batch_df.set_index(KEY_COLUMNS)

        batch_cache[batch_file] = batch_df

    else:
        batch_df = batch_cache[batch_file]

    # ----------------------------------------
    # Poišči isti SAMPLE + CONTIG NAME
    # ----------------------------------------

    key = (sample, contig)

    if key not in batch_df.index:

        print(
            f"NI NAJDEN: {sample} / {contig} "
            f"v {batch_file.name}"
        )

        not_found_contigs += 1
        continue

    source_row = batch_df.loc[key]

    # Če obstaja več identičnih SAMPLE + CONTIG NAME
    if isinstance(source_row, pd.DataFrame):

        print(
            f"OPOZORILO: več zadetkov za "
            f"{sample} / {contig} v {batch_file.name}. "
            f"Uporabljam prvega."
        )

        source_row = source_row.iloc[0]

    matched_contigs += 1

    # ========================================================
    # PRENESI MANJKAJOČE INFORMACIJE
    # ========================================================

    for column in source_row.index:

        source_value = source_row[column]

        # ----------------------------------------
        # Če source nima informacije, nima smisla
        # ničesar prenašati
        # ----------------------------------------

        if is_missing(source_value):
            continue

        # ----------------------------------------
        # Če stolpec sploh še ne obstaja v cv_3,
        # ga dodamo
        # ----------------------------------------

        if column not in df.columns:

            df[column] = pd.NA
            added_columns.add(column)

        # ----------------------------------------
        # Trenutna vrednost v cv_3
        # ----------------------------------------

        current_value = df.at[idx, column]

        # ----------------------------------------
        # Dopolni SAMO če informacija manjka
        # ----------------------------------------

        if is_missing(current_value):

            df.at[idx, column] = source_value
            filled_cells += 1


# ============================================================
# SHRANI
# ============================================================

df.to_excel(OUTPUT_FILE, index=False)


# ============================================================
# STATISTIKA
# ============================================================

print()
print("=" * 60)
print("KONČANO")
print("=" * 60)

print(f"Contigov v cv_3:       {len(df)}")
print(f"Najdenih v batchih:    {matched_contigs}")
print(f"Ni bilo najdenih:      {not_found_contigs}")
print(f"Dopolnjenih celic:      {filled_cells}")
print(f"Dodanih stolpcev:       {len(added_columns)}")

if added_columns:

    print("\nDodani stolpci:")

    for col in sorted(added_columns):
        print(f"  + {col}")


if missing_batch_files:

    print("\nManjkajoče batch datoteke:")

    for file in sorted(missing_batch_files):
        print(f"  - {file}")


print()
print(f"Rezultat: {OUTPUT_FILE}")