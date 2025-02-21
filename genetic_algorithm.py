import numpy as np
from typing import List, Tuple, Callable
import random

class GeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 elite_size: int = 5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def initialize_population(self, param_ranges: List[Tuple[float, float]]) -> np.ndarray:
        """Initialize random population within parameter ranges"""
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(low, high) for low, high in param_ranges]
            population.append(individual)
        return np.array(population)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents"""
        mask = np.random.random(len(parent1)) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(self, individual: np.ndarray, param_ranges: List[Tuple[float, float]]) -> np.ndarray:
        """Mutate individual genes with probability mutation_rate"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                low, high = param_ranges[i]
                mutated[i] = random.uniform(low, high)
        return mutated

    def evolve(self, 
               population: np.ndarray, 
               fitness_func: Callable,
               param_ranges: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve population for one generation"""
        # Calculate fitness for each individual
        fitness_scores = np.array([fitness_func(ind) for ind in population])

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        # Keep elite individuals
        new_population = population[:self.elite_size].copy()

        # Pre-allocate the rest of the population
        remaining_slots = self.population_size - self.elite_size
        offspring = np.zeros((remaining_slots, population.shape[1]))

        # Generate rest of population through selection and crossover
        for i in range(remaining_slots):
            # Tournament selection
            parent1_idx = random.randint(0, len(population)//2)
            parent2_idx = random.randint(0, len(population)//2)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover and mutation
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, param_ranges)
            offspring[i] = child

        # Combine elite individuals with offspring
        new_population = np.vstack([new_population, offspring])

        return new_population, fitness_scores[0]