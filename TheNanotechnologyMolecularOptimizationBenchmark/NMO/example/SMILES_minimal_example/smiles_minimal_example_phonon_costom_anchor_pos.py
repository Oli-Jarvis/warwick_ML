from NMO import Oracle_Handler_Smiles

oracle = Oracle_Handler_Smiles("./config_phonon.ini")

smiles = ["CC1=CC=C(C)C=C1", "C#CC#C"]

# Single pair applied to all molecules: atom 1 and atom 5 as anchors
fitness, rewards, _ = oracle.get_fitness(smiles, anchor_atoms=[0, 3])

print("Fitness:", fitness)
print("Rewards:", rewards)
# Expected output:
# Fitness: [0.         7.53752422]
# Rewards: defaultdict(<function Oracle_Handler.get_rewards.<locals>.<lambda> at 0x7fc751a06020>,
# {'smiles': array(['', '[Au]SC#CC#CS[Au]'], dtype='<U16'),
# 'SA': array([0.        , 5.86154593]),
# 'hash_values': array(['a20aab1e053b94e054f8792878828285', '932963deb37dc7fbbb306675bb93e002'], dtype='<U32'),
# 'hl_gaps': array([0.        , 1.66413959]),
# 'k_ph': array([0.        , 7.53752422]),
# 'failure_reasons': array(['Gold is connected to multiple atoms', ''], dtype='<U35'),
# 'oracle_calls': array([0, 1])})


smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1", "C1=CC=CC=C1"]
# Per-molecule anchor pairs
fitness, rewards, _ = oracle.get_fitness(smiles, anchor_atoms=[[0, 1], [0, 2], [0, 3]])

print("Fitness:", fitness)
print("Rewards:", rewards)
# Expected output:
# Fitness: [ 0.         10.87074375 11.52433777]
# Rewards: defaultdict(<function Oracle_Handler.get_rewards.<locals>.<lambda> at 0x7fc751a06520>,
# {'smiles': array(['', '[H]c1c([H])c(S[Au])c([H])c(S[Au])c1[H]', '[H]c1c([H])c(S[Au])c([H])c([H])c1S[Au]'], dtype='<U38'),
# 'SA': array([0.        , 2.9630412 , 2.94935868]), 'hash_values': array(['ca6cde9557168e845192faa4703c0349', '2100e15ed5986aa7f722bd5f65b39b80','addb05358834caf57276fb190db6ef37'], dtype='<U32'),
# 'hl_gaps': array([0.        , 1.94183403, 1.74579176]),
# 'k_ph': array([ 0.        , 10.87074375, 11.52433777]),
# 'failure_reasons': array(['Gold is connected to multiple atoms', '', ''], dtype='<U35'),
# 'oracle_calls': array([2, 3, 4])})

