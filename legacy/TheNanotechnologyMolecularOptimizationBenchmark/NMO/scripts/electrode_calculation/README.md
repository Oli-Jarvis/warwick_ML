# Utils

The directory contains the files first_dftb_in.hsd and second_dftb_in.hsd. They can be used (and adapted) to perform the dftb+ calculations. Two calculations have to be run because when matrices are written, dftb+ just write the data without scc. Rename the "detailed.out" of the first dftb+ calculation to "detailed_first.out" and name the log of the second dftb+ calculation "log.out" 

create_gold.py can be used to create the gold_structures and the POSCAR files. Creates files for left and right lead with mirror symmetry. Should be self explanatory.