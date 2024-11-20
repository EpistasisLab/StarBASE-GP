#####################################################################################################
#
# NSGA-II tool box for the selection and evolutionary process.
# Note that we are alwasy assuming objectives are to be maximized.
# As such, we expect that the values are converted to negative if they are to be minimized.
#
# Python 3.12.4: conda activate star-epi-pre
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Tuple
import numpy.typing as npt

# r2 score type
r2_t = np.float32
# feature count type
feature_cnt_t = np.int16
# traits type
traits_t = Tuple[r2_t, feature_cnt_t]
# numpy random number generator type
rng_t = np.random.Generator
# population id type
pop_id_t = np.uint16

@typechecked
def non_dominated_sorting(obj_scores: npt.NDArray) -> Tuple[List[npt.NDArray[np.uint16]],npt.NDArray[np.uint16]]:
    """
    Perform non-dominated sorting for a maximization problem using NumPy arrays of type float32.

    Parameters:
    obj_scores (np.ndarray): A 2D array where each row represents the objective values for a solution.

    Returns:
    Tuple(fronts, rank):
    fronts (list of numpy array of uint16): Each sublist contains the indices of solutions in the corresponding Pareto front.
    rank (numpy array of uint16): The front rank of each solution in the population.
    """
    # quick check to make sure that elements in scores are numpy arrays with np.float32
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure all elements are of the correct type
    assert all(isinstance(x[0], r2_t) for x in obj_scores)
    assert all(isinstance(x[1], feature_cnt_t) for x in obj_scores)
    # make sure all [0] elements are non-negative
    assert all(x[0] > 0.0 for x in obj_scores)
    assert all(x[1] < 0 for x in obj_scores)

    pop_size = obj_scores.shape[0]
    # final fronts returned
    fronts = [[]]
    # what front is solutions 'p' in
    rank = np.zeros(pop_size, dtype=np.uint16)
    # what 'q' solutions dominate 'p' solution
    domination_count = np.zeros(pop_size, dtype=np.uint16)
    # what 'q' solutions are dominated by 'p' solution
    dominated_solutions = [[] for _ in range(pop_size)]

    for p in range(pop_size):
        for q in range(pop_size):
            if dominates(obj_scores[p], obj_scores[q]):
                dominated_solutions[p].append(q)
            elif dominates(obj_scores[q], obj_scores[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                assert domination_count[q] >= 0 #check that it's always positive
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()

    fronts = [np.array(front, dtype=np.uint16) for front in fronts]
    return fronts, rank

@typechecked
def crowding_distance(obj_scores: npt.NDArray, count: np.int32) -> npt.NDArray[r2_t]:
    """
    Calculate the crowding distance for each individual in the population.

    Parameters:
    - obj_scores: List of performances on obj_scores for each individual. We are assuming that the
                position of scores are the same as the position of the individuals in the population.
    - count: Number of obj_scores.

    Returns:
    - crowding_distances: List of crowding distances corresponding to each individual.
    """

    # quick check to make sure that elements in scores are numpy arrays with np.float32
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure all elements are of the correct type
    assert all(isinstance(x[0], r2_t) for x in obj_scores)
    assert all(isinstance(x[1], feature_cnt_t) for x in obj_scores)
    # make sure all [0] elements are non-negative
    assert all(x[0] > 0.0 for x in obj_scores)
    assert all(x[1] > 0 for x in obj_scores)

    population_size = len(obj_scores)
    crowding_distances = np.zeros(population_size, dtype=np.float32)

    for m in range(count):
        # Sort the population based on the m-th objective
        sorted_indices = np.argsort([ind[m] for ind in obj_scores], kind='mergesort')
        sorted_population = obj_scores[sorted_indices]

        # Calculate the range of the m-th objective
        min_obj = sorted_population[0][m]
        max_obj = sorted_population[-1][m]

        # skip if both max and min are the same
        if max_obj == min_obj:
            continue

        # Set the crowding distance of boundary points to infinity
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf

        # Calculate crowding distances for intermediate points
        for i in range(1, population_size - 1):
            next_obj = sorted_population[i + 1][m]
            prev_obj = sorted_population[i - 1][m]
            crowding_distances[sorted_indices[i]] += (next_obj - prev_obj) / (max_obj - min_obj)

    return crowding_distances

@typechecked
def dominates(solution1: traits_t, solution2: traits_t) -> bool:
    """
    Check if solution1 dominates solution2.

    Parameters:
    solution1 (numpy array of np.float32): The first solution's objective values.
    solution2 (numpy array of np.float32): The second solution's objective values.

    Returns:
    bool: True if solution1 dominates solution2, False otherwise.
    """

    # check that solutions scores are of the same dimension
    assert len(solution1) == len(solution2)
    # make srue r2 are non-negative
    assert solution1[0] >= 0 and solution2[0] >= 0
    # make sure feature counts are negative
    assert solution1[1] < 0 and solution2[1] < 0
    # make sure they are the correct type
    assert isinstance(solution1[0], r2_t) and isinstance(solution2[0], r2_t)
    assert isinstance(solution1[1], feature_cnt_t) and isinstance(solution2[1], feature_cnt_t)

    greater_or_equal = solution1[0] >= solution2[0] and solution1[1] >= solution2[1]
    better_in_at_least_one = solution1[0] > solution2[0] or solution1[1] > solution2[1]

    return bool(greater_or_equal and better_in_at_least_one)

@typechecked
def non_dominated_binary_tournament(ranks: npt.NDArray[feature_cnt_t], distances: npt.NDArray[r2_t], rng_: rng_t) -> pop_id_t:

    # make sure that ranks and distances are the same size
    assert ranks.shape == distances.shape
    # initialize the random number generator
    rng = np.random.default_rng(rng_)

    # get two random number between 0 and the population size
    t1,t2 = rng.choice(len(ranks), size=2, replace=False)

    assert t1 != t2
    assert 0 <= t1 < len(ranks)
    assert 0 <= t2 < len(ranks)

    # check if the two solutions are in the same front
    if ranks[t1] == ranks[t2]:
        # the one with the greatest crowding distance wins
        return pop_id_t(t1) if distances[t1] > distances[t2] else pop_id_t(t2)

    # if they are in different fronts, the lower rank one wins
    else:
        return pop_id_t(t1) if ranks[t1] < ranks[t2] else pop_id_t(t2)

@typechecked
def non_dominated_truncate(fronts: List[npt.NDArray[feature_cnt_t]],
                           distances: npt.NDArray[r2_t],
                           N: pop_id_t) -> npt.NDArray[feature_cnt_t]:
    # make sure that fronts and distances are the nonempty
    assert sum([len(x) for x in fronts]) > 0 and len(distances) > 0
    # make sure that fronts and distances are the same size
    assert sum([len(x) for x in fronts]) == len(distances)
    # check that first object in fronts is a numpy array
    assert isinstance(fronts[0], np.ndarray)

    # surviving solutions
    survivors = []

    # go through each front and add the solutions to the survivors
    for front in fronts:
        # add solutions without ordering based on distance (as is)
        if len(survivors) + len(front) <= N:
            survivors.extend(front)
        else:
            # sort the front by crowding distance in decending order
            sorted_distance = np.flip(np.argsort(distances[front], kind='mergesort'))
            sorted_front = front[sorted_distance]
            survivors.extend(sorted_front[:N-len(survivors)])
            break

    assert len(survivors) == N
    return np.array(survivors, dtype=np.uint16)