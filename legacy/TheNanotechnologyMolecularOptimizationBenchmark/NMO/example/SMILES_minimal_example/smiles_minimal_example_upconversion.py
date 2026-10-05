from NMO import Oracle_Handler_Smiles

#To use the upconversion the enviroment variable XTB_PTB_BIN has to be set

oracle = Oracle_Handler_Smiles("./config_upconversion.ini")

smiles = ["C#CCCC#CCC#C", "C1=CC=CC=C1"]
fitness, rewards, oracle_calls_exceeded = oracle.get_fitness(smiles)

print("Fitness:", fitness)
print("Rewards:", rewards)
print("Oracle calls exceeded:", oracle_calls_exceeded)
