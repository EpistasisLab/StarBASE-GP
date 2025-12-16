#####################################################################################################
#
# NSGA-II tool box for the selection and evolutionary process.
# Note that we assume objectives are to be maximized, must convert to negative if needed.
#
#####################################################################################################

import numpy as np
from numba import njit
from typeguard import typechecked
from typing import List, Tuple
import numpy.typing as npt
from .types import (float32_t, int16_t, rng_t, uint32_t, int32_t, uint32_t)

@njit(cache=True, fastmath=True)
def _dominates_numba(sol1_r2, sol1_feat, sol2_r2, sol2_feat):
    """Numba-compiled dominance check."""
    greater_or_equal = sol1_r2 >= sol2_r2 and sol1_feat >= sol2_feat
    better_in_at_least_one = sol1_r2 > sol2_r2 or sol1_feat > sol2_feat
    return greater_or_equal and better_in_at_least_one

def _front_zero_core(r2_scores, feat_counts):
    """Core logic for front zero identification using Numba for dominance checks."""
    pop_size = len(r2_scores)
    front_zero = []
    
    for p in range(pop_size):
        dominated = False
        for q in range(pop_size):
            if q != p and _dominates_numba(r2_scores[q], feat_counts[q], r2_scores[p], feat_counts[p]):
                dominated = True
                break
        if not dominated:
            front_zero.append(p)
    
    return front_zero

def _non_dominated_sorting_core(r2_scores, feat_counts):
    """Core logic for non-dominated sorting using Numba for dominance checks."""
    pop_size = len(r2_scores)
    rank = np.zeros(pop_size, dtype=np.int32)
    domination_count = np.zeros(pop_size, dtype=np.int16)
    dominated_solutions = [[] for _ in range(pop_size)]
    fronts = [[]]
    
    for p in range(pop_size):
        for q in range(pop_size):
            if p != q:
                if _dominates_numba(r2_scores[p], feat_counts[p], r2_scores[q], feat_counts[q]):
                    dominated_solutions[p].append(q)
                elif _dominates_numba(r2_scores[q], feat_counts[q], r2_scores[p], feat_counts[p]):
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
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    if len(fronts[-1]) == 0:
        fronts.pop()
    
    return fronts, rank

@typechecked
def front_zero(obj_scores: npt.NDArray) -> List[int32_t]:
    """
    Identify the indices of solutions in the first Pareto front (non-dominated solutions)
    for a maximization problem using NumPy arrays of type float32.

    Parameters:
    obj_scores (np.ndarray): A 2D array where each row represents the objective values for a solution.

    Returns:
    np.ndarray: An array of indices corresponding to the non-dominated solutions.
    """

    # quick check to make sure that elements in scores are numpy arrays with float32_t
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure all elements are of the correct type
    assert all(isinstance(x[0], float32_t) for x in obj_scores)
    assert all(isinstance(x[1], int32_t) for x in obj_scores)
    # make sure all [0] elements are non-negative (maximization)
    assert all(x[0] > 0.0 for x in obj_scores)
    # make sure all [1] elements are negative (minimization)
    assert all(x[1] < 0 for x in obj_scores)

    # Extract arrays for Numba
    r2_scores = np.array([x[0] for x in obj_scores], dtype=np.float32)
    feat_counts = np.array([x[1] for x in obj_scores], dtype=np.int32)
    
    # Call Numba-compiled core function
    front_zero_list = _front_zero_core(r2_scores, feat_counts)
    
    return [int32_t(x) for x in front_zero_list]

@typechecked
def non_dominated_sorting(obj_scores: npt.NDArray) -> Tuple[List[npt.NDArray[int32_t]],npt.NDArray[int32_t]]:
    """
    Perform non-dominated sorting for a maximization problem using NumPy arrays of type float32.

    Parameters:
    obj_scores (np.ndarray): A 2D array where each row represents the objective values for a solution.

    Returns:
    Tuple(fronts, rank):
    fronts (list of numpy array): Each sublist contains the indices of solutions in the corresponding Pareto front.
    rank (numpy array of int32_t): The front rank of each solution in the population.
    """

    # quick check to make sure that elements in scores are numpy arrays with float32_t
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure all elements are of the correct type
    assert all(isinstance(x[0], float32_t) for x in obj_scores)
    assert all(isinstance(x[1], int32_t) for x in obj_scores)
    # make sure all [0] elements are non-negative (maximization)
    assert all(x[0] > 0.0 for x in obj_scores)
    # make sure all [1] elements are negative (minimization)
    assert all(x[1] < 0 for x in obj_scores)

    # Extract arrays for Numba
    r2_scores = np.array([x[0] for x in obj_scores], dtype=np.float32)
    feat_counts = np.array([x[1] for x in obj_scores], dtype=np.int32)
    
    # Call Numba-compiled core function
    fronts_list, rank = _non_dominated_sorting_core(r2_scores, feat_counts)
    
    # Convert to numpy arrays
    fronts = [np.array(front, dtype=int32_t) for front in fronts_list]
    return fronts, rank.astype(int32_t)

@typechecked
def crowding_distance(obj_scores: npt.NDArray, front_map, count = int16_t(2)) -> npt.NDArray[float32_t]:
    """
    Calculate the crowding distance for each individual in the population.

    Parameters:
    - obj_scores: List of performances on obj_scores for each individual. We are assuming that the
                position of scores are the same as the position of the individuals in the population.
    - count: Number of obj_scores.

    Returns:
    - crowding_distances: List of crowding distances corresponding to each individual.
    """

    # quick check to make sure that elements in scores are numpy arrays with float32_t
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure all elements are of the correct type
    assert all(isinstance(x[0], float32_t) for x in obj_scores)
    assert all(isinstance(x[1], int32_t) for x in obj_scores)
    # make sure all [0] elements are non-negative
    assert all(x[0] > 0.0 for x in obj_scores)
    assert all(x[1] > 0 for x in obj_scores)

    # initialize the crowding distances to negative for guards
    crowding_distances = np.full(len(obj_scores), float32_t(-1.0), dtype=float32_t)

    for front in front_map:
        # set inital front crowding distances to zero for addition
        crowding_distances[front] = float32_t(0.0)

        for m in range(count):
            # Sort the front scores based on the m-th objective
            sorted_indices = np.argsort([ind[m] for ind in obj_scores[front]], kind='mergesort')
            sorted_front = obj_scores[front[sorted_indices]]

            # calculate the range of the m-th objective
            min_obj = sorted_front[0][m]
            max_obj = sorted_front[-1][m]

            # skip if both max and min are the same
            if max_obj == min_obj:
                continue

            # set the crowding distance of boundary points to infinity
            crowding_distances[front[sorted_indices[0]]] = np.inf
            crowding_distances[front[sorted_indices[-1]]] = np.inf

            # calculate crowding distances for intermediate points
            for i in range(1, len(front) - 1):
                next_obj = sorted_front[i + 1][m]
                prev_obj = sorted_front[i - 1][m]
                crowding_distances[front[sorted_indices[i]]] += float32_t(next_obj - prev_obj) / float32_t(max_obj - min_obj)

    # make sure all crowding distances are non-negative
    assert np.all(crowding_distances >= 0.0)

    return crowding_distances

@typechecked
def dominates(solution1: Tuple[float32_t, int32_t], solution2: Tuple[float32_t, int32_t]) -> bool:
    """
    Check if solution1 dominates solution2.

    Parameters:
    solution1 (Tuple[float32_t, int32_t]): The first solution's objective values.
    solution2 (Tuple[float32_t, int32_t]): The second solution's objective values.

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
    assert isinstance(solution1[0], float32_t) and isinstance(solution2[0], float32_t)
    assert isinstance(solution1[1], int32_t) and isinstance(solution2[1], int32_t)

    greater_or_equal = solution1[0] >= solution2[0] and solution1[1] >= solution2[1]
    better_in_at_least_one = solution1[0] > solution2[0] or solution1[1] > solution2[1]

    return bool(greater_or_equal and better_in_at_least_one)

@typechecked
def non_dominated_binary_tournament(ranks: npt.NDArray[int32_t], distances: npt.NDArray[float32_t], rng: rng_t) -> uint32_t:
    """
    Perform a binary tournament selection based on non-dominated sorting and crowding distance.
    First, two individuals are randomly selected from the population.
    Winners are determined based on their ranks (fronts) and crowding distances.
    Lower rank individuals are preferred, followed by higher crowding distances in case of ties.

    Args:
        ranks (npt.NDArray[int32_t]): The front rank of each solution in the population.
        distances (npt.NDArray[float32_t]): The crowding distance of each solution in the population.
        rng (rng_t): Random number generator.

    Returns:
        uint32_t: The index of the winning individual.
    """

    # make sure that ranks and distances are the same size
    assert ranks.shape == distances.shape

    # get two random number between 0 and the population size
    t1,t2 = rng.choice(len(ranks), size=2, replace=False)
    t1, t2 = uint32_t(t1), uint32_t(t2)

    assert t1 != t2
    assert 0 <= t1 < len(ranks)
    assert 0 <= t2 < len(ranks)

    # check if the two solutions are in the same front
    if ranks[t1] == ranks[t2]:
        # the one with the greatest crowding distance wins
        return t1 if distances[t1] > distances[t2] else t2

    # if they are in different fronts, the lower rank one wins
    else:
        return t1 if ranks[t1] < ranks[t2] else t2

@typechecked
def non_dominated_truncate(fronts: List[npt.NDArray[int16_t]], distances: npt.NDArray[float32_t], N: int16_t) -> npt.NDArray[int16_t]:
    """
    Truncate the population to the N best individuals based on non-dominated sorting and crowding distance.
    First, individuals are added front by front until adding another front would exceed N.
    If the last front cannot be fully added, individuals from that front are selected based on their
    crowding distances in descending order.

    Args:
        fronts (List[npt.NDArray[int16_t]]): List of Pareto fronts, each containing indices of individuals.
        distances (npt.NDArray[float32_t]): The crowding distance of each solution in the population.
        N (int16_t): The desired population size after truncation.

    Returns:
        npt.NDArray[int16_t]: Indices of the selected individuals after truncation.
    """

    # make sure that fronts and distances are the same size
    assert sum([len(x) for x in fronts]) == len(distances)
    # make sure each front in the list are within the correct range
    assert all(all(0 <= ind < len(distances) for ind in front) for front in fronts)
    # make sure that distances is non-empty
    assert len(distances) > 0
    # make sure that distances are non-negative
    assert np.all(distances >= 0.0)
    # check that first object in fronts is a numpy array
    assert isinstance(fronts[0], np.ndarray)

    # go through each front and add the solutions to the survivors
    survivors = []
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

    # make sure all survivor ids are within the correct range
    assert all(0 <= s < len(distances) for s in survivors)
    return np.array(survivors, dtype=int16_t)