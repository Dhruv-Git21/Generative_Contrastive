import random
from models.rga.scoring import embedding_similarity

def select_by_fitness(population, fitness_scores, num_parents=10):
    # Select the top num_parents molecules based on fitness (higher is better)
    sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
    parents_idx = sorted_indices[:num_parents]
    return [population[i] for i in parents_idx]

def crossover(mom, dad):
    # Simple crossover: take first half of mom's SMILES and second half of dad's (very naive, often invalid)
    from rdkit import Chem
    smi_mom = Chem.MolToSmiles(mom) if mom else ""
    smi_dad = Chem.MolToSmiles(dad) if dad else ""
    if not smi_mom or not smi_dad:
        return mom, dad
    cut1 = len(smi_mom)//2
    cut2 = len(smi_dad)//2
    child1_smi = smi_mom[:cut1] + smi_dad[cut2:]
    child2_smi = smi_dad[:cut2] + smi_mom[cut1:]
    try:
        from rdkit import Chem
        child1 = Chem.MolFromSmiles(child1_smi)
        child2 = Chem.MolFromSmiles(child2_smi)
    except:
        child1 = None
        child2 = None
    return child1 if child1 else mom, child2 if child2 else dad

def mutate(mol, guidance=None):
    # Simple mutation: random atom substitution (change one atom to a different element)
    from rdkit import Chem
    if mol is None:
        return None
    mol = Chem.RWMol(mol)
    atom_count = mol.GetNumAtoms()
    if atom_count == 0:
        return mol.GetMol()
    idx = random.randrange(atom_count)
    atom = mol.GetAtomWithIdx(idx)
    current_symbol = atom.GetSymbol()
    # Define a small set of possible atoms
    possible_atoms = ["C", "N", "O", "F", "S"]
    if current_symbol in possible_atoms:
        possible_atoms.remove(current_symbol)
    new_symbol = random.choice(possible_atoms) if possible_atoms else current_symbol
    atom.SetAtomicNum(Chem.Atom(new_symbol).GetAtomicNum())
    try:
        new_mol = mol.GetMol()
        Chem.SanitizeMol(new_mol)
    except Chem.SanitizeException:
        new_mol = mol.GetMol()
    return new_mol

def genetic_optimize(initial_population, target_embed=None, generations=50):
    """
    Run a genetic algorithm to optimize molecules towards the target embedding (or other fitness).
    """
    population = initial_population.copy()
    best_mol = None
    best_score = -float('inf')
    for gen in range(generations):
        # Compute fitness for each molecule
        fitness_scores = []
        for mol in population:
            score = embedding_similarity(mol, target_embed)
            fitness_scores.append(score)
            if score > best_score:
                best_score = score
                best_mol = mol
        # Select parents
        parents = select_by_fitness(population, fitness_scores, num_parents=min(10, len(population)))
        # Generate offspring by crossover
        offspring = []
        for i in range(0, len(parents)-1, 2):
            mom = parents[i]
            dad = parents[i+1]
            child1, child2 = crossover(mom, dad)
            offspring += [child1, child2]
        # Mutate some offspring
        new_population = parents[:]  # elitism: carry parents over
        for child in offspring:
            if child is None:
                continue
            if random.random() < 0.3:
                child = mutate(child, guidance=target_embed)
            if child:
                new_population.append(child)
        # Refill population if it shrank
        while len(new_population) < len(population):
            new_population.append(random.choice(parents))
        population = new_population[:len(population)]
    # Return the best molecule found
    return best_mol
