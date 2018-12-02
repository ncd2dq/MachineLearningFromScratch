import random
from typing import TypeVar, List
import os
import time

DNAType = TypeVar('DNAType', bound='DNA')
SentenceType = TypeVar('SentenceType', bound='Sentence')

class DNA(object):
    gene_pool = 'abcdefghijklmnopqrstuvwxyz ?!,'

    def __init__(self, sent_length: int, genes: str=None) -> None:
        if genes is None:
            self.genes = self.create_genes(sent_length)
        else:
            self.genes = genes

    def create_genes(self, sent_length: int) -> str:
        '''
        Randomly select characters to create initial gene set
        '''
        genes = ''.join([random.choice(DNA.gene_pool) for _ in range(sent_length)])

        return genes

    def cross_over(self, other: DNAType, cross_over_percent: float=0.5) -> DNAType:
        '''
        Randomly swaps genes with another piece of DNA

        '''
        new_gene_sequence = ''.join([self.genes[i] if random.random() > cross_over_percent else other.genes[i] for i in range(len(self.genes))])

        return DNA(len(self.genes), new_gene_sequence)

    def mutate(self) -> None:
        '''
        Randomly swaps out one strand in the gene list inplace
        '''
        rand_index = random.randint(0, len(self.genes) - 1)
        temp_list = list(self.genes)
        temp_list[rand_index] = random.choice(DNA.gene_pool)
        self.genes = ''.join(temp_list)

    def __str__(self):
        return self.genes

    def __len__(self):
        return len(self.genes)


class Sentence(object):
    def __init__(self, sent_length: int, dna: DNAType=None) -> None:
        if dna is None:
            self.dna = DNA(sent_length)
        else:
            self.dna = dna
        self.fitness = 0

    def __len__(self):
        return len(self.dna)

    def __str__(self):
        return self.dna.__str__()

    def __add__(self, other: SentenceType) -> SentenceType:
        '''
        Execute the cross over method of the DNA class on each
        '''
        new_dna = self.dna.cross_over(other.dna)
        return Sentence(len(self.dna), new_dna)

    def calculate_fitness(self, target: str) -> None:
        '''
        Compare each element in dna.gene sequence to the target
        Square of fitness done in place
        '''
        fitness = 0
        for my_sequence, target_sequence in zip(self.dna.genes, target):
            if my_sequence == target_sequence:
                fitness += 1
        fitness *= fitness
        self.fitness = fitness

    def mutate(self, mutation_rate: float) -> None:

        if random.random() < mutation_rate:
            self.dna.mutate()


class Population(object):

    def __init__(self, pop_size: int, dna_length: int, mutation_rate: float=0.08) -> None:
        self.population = self.create_random_population(dna_length, pop_size)
        self.mutation_rate = mutation_rate

    def create_random_population(self, dna_length: int, pop_size: int) -> List[SentenceType]:
        new_pop = [Sentence(dna_length) for _ in range(pop_size)]

        return new_pop

    def selection(self, target: str) -> None:
        '''
        Have all children calculate fitness
        Normalize fitness scores
        '''
        for entity in self.population:
            entity.calculate_fitness(target)

        # Normalize fitness scores
        self.population = sorted(self.population, key=lambda elm: elm.fitness)
        max_fitness = self.population[-1].fitness
        for index, entity in enumerate(self.population):
            self.population[index].fitness = entity.fitness / max_fitness


    def crossover(self) -> List[SentenceType]:
        '''
        Choose two parents from population based on fitness and have them mate
        ''' 
        new_population = []
        while len(new_population) < len(self.population):
            parent_a = None
            parent_b = None
            while parent_a is None:
                potential_parent = random.choice(self.population)
                if random.random() < potential_parent.fitness:
                    parent_a = potential_parent
            while parent_b is None and parent_b is not parent_a:
                potential_parent = random.choice(self.population)
                if random.random() < potential_parent.fitness:
                    parent_b = potential_parent

            new_population.append(parent_a + parent_b)

        return new_population

    def mutation(self, mutation_rate: float) -> None:
        for entity in self.population:
            entity.mutate(mutation_rate)

    def get_best_child(self) -> SentenceType:
        return self.population[-1]

    def total_fitness(self) -> int:
        return sum(entity.fitness for entity in self.population)

    def percent_learned(self, best: SentenceType, target: str) -> float:
        total_possible = len(target)
        total = 0
        for b, t in zip(best.dna.genes, target):
            if b == t:
                total += 1
        return total / total_possible

    def life_cycle(self, epocs, target: str) -> None:

        for i in range(epocs):
            self.selection(target)
            best = self.get_best_child()
            self.population = self.crossover()
            self.mutation(self.mutation_rate)

            os.system('cls')
            print('\rCurrent Generation: {}'.format(i))
            print('\rTarget: {}'.format(target))
            print('\rBest  : {}'.format(best))
            print('\rPercent Complete: {}%'.format(self.percent_learned(best, target)*100))
            if best.dna.genes == target:
                print('DONE')
                break


target = 'im learning based on selection, crossover, and mutation'
p = Population(10000, len(target))
p.life_cycle(15000, target)
input('------End of lifecycle')