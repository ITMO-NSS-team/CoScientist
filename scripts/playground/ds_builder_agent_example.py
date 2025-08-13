from ChemCoScientist.agents.agents import dataset_builder_agent


def main():
    """
    Iterates through a list of chemical property queries, uses an agent to search for molecules matching those properties, and saves the resulting molecular data to a CSV file.
    
    Args:
        None
    
    Returns:
        None
    """
    querys = [
        """
        Find molecules that contain all following properties:
        1) Smiles
        2) Molecular Weight (between 250 and 500 Da)
        3) AlogP (between -2 and 5)
        4) Polar Surface Area (PSA) (between 20 and 150 Å²)
        5) #RO5 Violations (exactly 0 or 1)
        6) CX LogP (between -1 and 6)
        7) Aromatic Rings (between 0 and 6)
        8) Heavy Atoms (between 15 and 20)
        9) Molecular Formula
        """,
        "Connections with a number of rotatable bonds of no more than 5 and a positive LogD are required.",
        "Find molecules with positive LogD.",
        "Molecules with a polar surface area (PSA) of less than 100 are needed",
    ]

    for q in querys:
        print("========")
        print("Query:", q)
        res = dataset_builder_agent({"pending_tasks": [q], "responses": []})
        res["responses"][0].to_csv('./ChemCoScientist/dataset_handler/chembl/test.csv')
        print(res["responses"][0])
        print(res["responses"][0].iloc[0])


if __name__ == "__main__":
    main()