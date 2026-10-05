from rdkit import Chem

# Forbidden smarts patterns
BAD_PATTERNS = {
    "Poly_yne_Unstable": "C#CC#CC#C",
    "Cumulene_Unstable": "C=C=C=C",
    "Peroxide": "[OX2,OX1]-[OX2,OX1]",
    "Hydrazine": "[N&!R]-[N&!R]",
    "Anhydride_Instability": "[#6](=O)-[#8]-[#6](=O)",
    "Acyl_Urea_Instability": "[#6](=O)-[#7]-[#6](=O)",
    "Heteroatom_Alkyne": "[#8,#7]-[#6]#[#6]",
    "Acid_Halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Geminal_Heteroatoms": "[CX4](-[O,N,S,F,Cl,Br,I])(-[O,N,S,F,Cl,Br,I])",
    "N_or_O_Halogen": "[#7,#8]-[F,Cl,Br,I]",

    # Explosives (Nitramines/Nitrates)
    # Catches N-NO2 and O-NO2
    "Nitro_Instability": "[#7,#8]-[#7+](=O)[O-]",

    # Phosphorus Reactivity
    # Catches P-Cl, P-F (Instant hydrolysis)
    "Phosphorus_Halogen": "[#15]-[F,Cl,Br,I]",
    # Catches P-O-P (Unstable Anhydride)
    "Phospho_Anhydride": "[#15]-[#8]-[#15]",

    # Reactive Acylating Agents
    # Catches O=C-CN (Acyl Cyanide)
    "Acyl_Cyanide": "[CX3](=O)-[#6]#[#7]",

    # Unstable Imine Derivatives
    "Imidoyl_Halide": "[F,Cl,Br,I]-[#6]=[#7]",

    # Reactive Alkylators
    # Catches O=C-C-Cl (Alpha-halo carbonyl)
    "Alpha_Halo_Carbonyl": "[CX3](=O)-[CX4]-[Cl,Br,I]",

    "Diazonium": "[N+]#[N-]",

    "Unstable_Nitroso": "[!c]-N=O",

    "Ketene": "[#6]=[#6]=[#8]",

    "Ketenimine": "[#6]=[#6]=[#7]",

    "Carbodiimide": "[#7]=[#6]=[#7]",

    "Unstable_Azo": "[!c]-[#7]=[#7]",

    "Unstable_Pentalene": "[C&R2&r5&!r6&!r7&^2]@[C&R2&r5&!r6&!r7&^2]",

    "Unstable_Enamine_H": "[#7;H1,H2]-[#6;!a]=[#6]",

    "Strained_3_Ring": "[r3]",

    "Strained_4_Ring": "[r4]",

    "Impossible_Cyclic_Alkyne": "[#6]#[#6;R&r3,r4,r5,r6,r7]",

    "Unstable_N_S": "[#7]-[#16]",

    "Unstable_Terminal_Alkene": "[#6;R]=[#6;D1]",

    "Unstable_Quinoid_General": "[#6]=[c;R]([c;R])"


}

COMPILED_PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in BAD_PATTERNS.items()}


def validate_molecule(mol_input):
    """
    Checks a molecule (SMILES or RDKit Mol) against the forbidden patterns.
    Returns: (bool is_safe, list reason)
    """
    # Handle input type (SMILES string vs Mol object)
    if isinstance(mol_input, str):
        mol = Chem.MolFromSmiles(mol_input)
        if mol is None:
            return False, ["Invalid SMILES"]
    else:
        mol = mol_input

    issues = []

    # Check against all bad patterns
    for name, pattern in COMPILED_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            issues.append(name)

    is_safe = (len(issues) == 0)
    return is_safe, issues

if __name__ == "__main__":


    # test examples
    examples = [
        ("Safe_OPE_Diyne", "Sc1ccc(C#CC#Cc2ccccc2)cc1"),  # 2 Triple bonds (Safe)
        ("Unsafe_Tetrayne", "c1ccccc1C#CC#CC#CC#Cc1ccccc1"),  # 4 Triple bonds (BAD)
        ("Safe_Amide", "O=C(N)c1ccccc1"),  # Amide (Safe)
        ("Unsafe_AcidChloride", "O=C(Cl)c1ccccc1"),  # Acid Chloride (BAD)
        ("Unsafe_Peroxide", "COc1ccc(OOC)cc1"),  # Peroxide (BAD)
        ("Unsafe_Hemiaminal", "CC(O)(N)C"),  # Geminal OH and NH2 (BAD)
        ("Unsafe Nitroso", "[H]c1c([H])c(=C(N=O)C([H])([H])[H])c(Br)c([H])c1=C(N=O)C([H])([H])S[Au]"),
        ("Unsafe Cumulene", "[H]N=C=C=C=C=C=C=C=C(C(=C(C1=C([H])C(=NS[Au])N=N1)N([H])[H])N([H])[H])C([H])([H])[H]"),
        ("Safe Azobenzene", "c1ccccc1-N=N-c1ccccc1"),
        ("bad azo group", "C-N=N-C"),
        ("Pentalene", "C1=CC2=CC=CC2=C1"),
        ("Safe Bicyclooctane", "[H][C@@]12CCC[C@@]1([H])CCC2"),
        ("Safe Azulene", "c1cccc2cccc2c1"),
        ("unsafe azo group", "[H]C#CC(=C(N=C(N=C(N=NC(=NC(=C(C#CS[Au])C([H])([H])[H])N([H])[H])N([H])[H])N([H])[H])N([H])[H])N([H])[H])N([H])[H]"),
        ("unsafe Enamine_H", "[H]C1=C(Cl)C(C(=C2C([H])=C(N([H])[H])C(F)=C(C3=C(F)C(N([H])[H])=C3N([H])[H])C2([H])[H])N([H])[H])=C(Br)C1=NS[Au]"),
        ("safe Enamine_H", "C1CCC(=CC1)N2CCCC2"),
        ("strained 4 ring", "[H]OC1=C(C2=C(C3=NC(=NS[Au])C([H])=C(O[H])C3([H])[H])N=C2[H])C(Cl)=C1[H]"),
        ("strained 4 ring2", "[H]OC1=C([H])C(=NS[Au])N=C(C2=CC([H])=N2)C1([H])[H]"),
        ("impossible cyclic alkyne", "[H]OC(=C([H])C1=C(C([H])=C([H])C2=C(C#CS[Au])C2([H])[H])C#CC1([H])[H])N([H])[H]"),
        ("test", "[H]C1=C(Cl)C(C(=C2C([H])=C(Br)C(F)=C(C3=C(F)C(N([H])[H])=C3N([H])[H])C2([H])[H])N([H])[H])=C(Br)C1=NS[Au]"),
        ("unstable terminal alkene", "[H]C(S[Au])=C([H])C([H])=c1c([H])c([H])c(=c2c([H])c([H])c(=c3c([H])c([H])c(=C([H])[H])c([H])c3[H])c(Cl)c2[H])c([H])c1[H]"),
        ("unstable quinoid", "[H]C(S[Au])=C([H])C([H])=c1c([H])c([H])c(=c2c([H])c([H])c(=c3c([H])c([H])c(=C([H])[H])c([H])c3[H])c(Cl)c2[H])c([H])c1[H]"),
        ("empty", "")
    ]

    print(f"{'Name':<20} | {'Status':<8} | {'Reason'}")
    print("-" * 60)

    for name, smi in examples:
        safe, reasons = validate_molecule(smi)
        status = "KEEP" if safe else "DROP"
        reason_str = ", ".join(reasons) if reasons else "-"
        print(f"{name:<20} | {status:<8} | {reason_str}")