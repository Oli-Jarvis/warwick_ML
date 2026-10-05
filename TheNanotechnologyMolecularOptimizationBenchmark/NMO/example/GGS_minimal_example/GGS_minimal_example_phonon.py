from NMO import Oracle_Handler_GGS

oracle = Oracle_Handler_GGS("./config.ini")

ggs_strings = ["[:1frag_57][C][:0frag_17][pop][#Branch]",
          "[:0frag_41][Ring1][:0frag_0][Ring2]"]
fitness, rewards, oracle_calls_exceeded = oracle.get_fitness(ggs_strings)

print("Fitness:", fitness)
print("Rewards:", rewards)
print("Oracle calls exceeded:", oracle_calls_exceeded)
