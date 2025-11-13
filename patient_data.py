import pandas as pd


def load_tcga_patient_data(
    patients_info_path="dataset/tcga/patients_info.csv.gz",
    gene_exp_path="dataset/tcga/gene_exp.csv.gz",
    nsc_cid_smiles_path="Aug8_nsc_cid_smiles.csv",
    name2smiles_path="name2smiles.txt.gz",
):
    def to_parent(name):
        replacements = [
            " Hydrochloride",
            " Acetate",
            " Disodium",
            " Sodium",
            " Tartrate",
            " Citrate",
            " Pamoate",
            " Phosphate",
            " Tosylate",
            " Mesylate",
            " Poliglumex",
        ]
        for rpl in replacements:
            if name.endswith(rpl):
                return name.replace(rpl, "")
        return name

    # Load patient and gene expression data
    patients = pd.read_csv(patients_info_path, index_col=0)
    gene_exp = pd.read_csv(gene_exp_path, index_col=0)
    df = patients.merge(gene_exp)
    exp = df.iloc[:, 6:]

    # Prepare treatment DataFrame
    j = pd.DataFrame({"treatment": df["treatments.therapeutic_agents"].unique()})
    j["parent"] = j["treatment"].apply(to_parent)

    # Load SMILES data
    r = pd.read_csv(nsc_cid_smiles_path, index_col=0)[["NAME", "SMILES"]].dropna()
    tmp = pd.read_csv(name2smiles_path, sep="\t", header=None).drop_duplicates()
    tmp.columns = ["NAME", "SMILES"]
    data = {
        "Taxane Compound": "C[C@@H]1CCC[C@@]2([C@@H]1C[C@@H]3CC[C@H]([C@@H](C3(C)C)CC2)C)C",
        "Gamma-Secretase Inhibitor RO4929097": "CC(C)(C(=O)NCC(C(F)(F)F)(F)F)C(=O)N[C@H]1C2=CC=CC=C2C3=CC=CC=C3NC1=O",
        "Akt Inhibitor MK2206": "C1CC(C1)(C2=CC=C(C=C2)C3=C(C=C4C(=N3)C=CN5C4=NNC5=O)C6=CC=CC=C6)N",
        "Pan-VEGFR/TIE2 Tyrosine Kinase Inhibitor CEP-11981": "CC(C)CN1C2=C(C=C(C=C2)NC3=NC=CC=N3)C4=C1C5=C(C6=CN(N=C6CC5)C)C7=C4CNC7=O",
        "Carmustine Implant": "C(CCl)NC(=O)N(CCCl)N=O",
        "Aurora Kinase/VEGFR2 Inhibitor CYC116": "CC1=C(SC(=N1)N)C2=NC(=NC=C2)NC3=CC=C(C=C3)N4CCOCC4",
    }
    q = pd.DataFrame(list(data.items()), columns=["NAME", "SMILES"])
    r = pd.concat([r, tmp, q])

    # Merge treatment and SMILES
    j = j.merge(r, left_on="parent", right_on="NAME", how="left")

    # Merge with main df
    df_merged = (
        df.merge(
            j[["treatment", "SMILES"]],
            left_on="treatments.therapeutic_agents",
            right_on="treatment",
            how="left",
        )
        .dropna()
        .drop_duplicates()
    )
    return df_merged, exp
