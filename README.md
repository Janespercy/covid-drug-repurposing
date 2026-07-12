# Assignment 1

# Class Info
- Subject: BINF6200 Bioinformatics Programming
- Professor: Chesley Leslin
- Section: 06

# Files
1. protein_to_daltons.py
2. input_to_amino_acids.py
3. input_to_protocol.py
 
## Author: Janes Percy Johnson

# Assignment Description
## Protein and Solution Preparation Programs:

This repository contains three Python programs that perform simple molecular biology calculations:

1. **protein_to_daltons.py**:
   - Hard-coded protein length of 671 amino acids.
   - Calculates the estimated molecular weight in kilodaltons (kDa) based on an average molecular weight of 110 Daltons per amino acid.
   
2. **input_to_amino_acids.py**:
   - Asks the user for a gene name and the number of nucleotides in the DNA sequence.
   - Calculates the number of amino acids in the resulting protein and its estimated molecular weight.
   - Validates that the DNA sequence length is divisible by 3.
   
3. **input_to_protocol.py**:
   - Takes inputs from the user regarding NaCl and MgCl2 concentrations and the final volume of the solution.
   - Calculates the amount of each reagent to add based on user input.

# Programs

## Program -1 (protein_to_daltons.py)

"""Hard-coded sequence of Rattus norvegicus PKC Beta-1 protein"""

def main():
    protein_sequence="""MADPAAGPPPSEGEESTVRFARKGALRQKNVHEVKNHKFTARFFKQPTFCSHCTDFIWGFGKQGFQCQVC
CFVVHKRCHEFVTFSCPGADKGPASDDPRSKHKFKIHTYSSPTFCDHCGSLLYGLIHQGMKCDTCMMNVH
KRCVMNVPSLCGTDHTERRGRIYIQAHIDREVLIVVVRDAKNLVPMDPNGLSDPYVKLKLIPDPKSESKQ
KTKTIKCSLNPEWNETFRFQLKESDKDRRLSVEIWDWDLTSRNDFMGSLSFGISELQKAGVDGWFKLLSQ
EEGEYFNVPVPPEGSEGNEELRQKFERAKIGQGTKAPEEKTANTISKFDNNGNRDRMKLTDFNFLMVLGK
GSFGKVMLSERKGTDELYAVKILKKDVVIQDDDVECTMVEKRVLALPGKPPFLTQLHSCFQTMDRLYFVM
EYVNGGDLMYHIQQVGRFKEPHAVFYAAEIAIGLFFLQSKGIIYRDLKLDNVMLDSEGHIKIADFGMCKE
NIWDGVTTKTFCGTPDYIAPEIIAYQPYGKSVDWWAFGVLLYEMLAGQAPFEGEDEDELFQSIMEHNVAY
PKSMSKEAVAICKGLMTKHPGKRLGCGPEGERDIKEHAFFRYIDWEKLERKEIQPPYKPKARDKRDTSNF
DKEFTRQPVELTPTDKLFIMNLDQNEFAGFSYTNPEFVINV"""

    # To get rid of new line characters
    protein_sequence=protein_sequence.replace('\r', '').replace('\n', '')

    # Average molecular weight per amino acid (in Daltons)
    average_molecular_weight=110

    # Calculate protein length
    protein_length=len(protein_sequence) #Number of amino acids in the protein
    print(f"The length of 'Protein kinase C beta type' is: {protein_length} ")

    #Calculate molecular weight in Daltons
    molecular_weight_daltons = protein_length * average_molecular_weight

    #Convert to Kilodaltons
    molecular_weight_kilodaltons =molecular_weight_daltons / 1000

    # Print estimated molecular weight
    print(f"The average weight of this protein in kilodaltons is: {molecular_weight_kilodaltons} ")

main()

## Program -2 (input_to_amino_acids.py):

import sys

def main():
    gene_name = input("Please a name for the DNA sequence: ")
    print(f"Your sequence name is: {gene_name}")

    while True:
        try:
            number_nucleotides = float(input("Please enter the length of the sequence:"))
            if number_nucleotides <= 0:
                print("Please enter a positive number:")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a positive number.")

    print(f"The length of the DNA sequence is: {number_nucleotides}")

    if number_nucleotides % 3!=0:
        print("\n\nError: the DNA sequence is not a multiple of 3")
        sys.exit(1)
    else:
        number_amino_acids = number_nucleotides / 3
        print(f"The length of the decoded protein is: {number_amino_acids}")
        estimated_molecular_weight = (number_amino_acids * 110) / 1000
        print(f"The average weight of the protein sequence is: {estimated_molecular_weight}")

main()

## Program -3 (input_to_protocol.py):

def main():
    """Business logic"""
    final_vol = float(input("Please enter the final volume of the solution (ml): "))

    # NaCl
    nacl_stock = float(input("Please enter the NaCl stock (mM): "))
    nacl_final = float(input("Please enter the NaCl final (mM): "))

    # MgCl2
    mg_stock = float(input("Please enter the MgCl2 stock (mM): "))
    mg_final = float(input("Please enter the MgCl2 final (mM): "))

    # Calculate volumes
    nacl_volume = final_vol * (nacl_final / nacl_stock)
    mg_volume = final_vol * (mg_final / mg_stock)

    # Print protocol
    print(f"Add {nacl_volume:.3f} ml NaCl")
    print(f"Add {mg_volume:.3f} ml MgCl2")
    print(f"Add water to a final volume of {final_vol:.1f} ml and mix")

if __name__ == '__main__':
    main()


## How to Run the Programs

1. Clone the repository or download the files.
2. Run the programs in a Python 3 environment using the following commands:

   ```bash
   $ python3 protein_to_daltons.py
   $ python3 input_to_amino_acids.py
   $ python3 input_to_protocol.py
   
3. Follow the on-screen prompts to input values and calculate results.


# Dependencies
Python 3.11

# License
MIT License
