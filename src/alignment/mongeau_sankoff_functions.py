import numpy as np

# Mapping semitone differences to degrees
SEMITONE_TO_DEGREE = {
    0: 0,  # Unison
    1: 1,  # Minor second
    2: 2,  # Major second
    3: 2,  # Minor third
    4: 3,  # Major third
    5: 4,  # Perfect fourth
    6: 5,  # Tritone
    7: 4,  # Perfect fifth
    8: 5,  # Minor sixth
    9: 5,  # Major sixth
    10: 6, # Minor seventh
    11: 6  # Major seventh
}

# Mapping degrees to weights
DEGREE_WEIGHTS = {
    0: 0.0,   # Unison (identity replacement), octave, two octaves
    4: 0.1,   # Fifth, octave and a fifth
    2: 0.2,   # Third, octave and a third
    5: 0.35,  # Sixth
    3: 0.5,   # Fourth
    6: 0.8,   # Seventh
    1: 0.9    # Second
}

# Function to calculate degree weight based on interval
def get_degree_weight(interval):
    """
    Returns the weight for a given interval based on predefined degree weights.
    """
    # Step 1: Convert semitones to degrees
    degree = SEMITONE_TO_DEGREE.get(abs(interval) % 12, None)
    
    # Step 2: Get the weight for that degree, default to 2.6 for dissonant intervals if not found
    if degree is not None:
        return DEGREE_WEIGHTS.get(degree, 2.6)
    else:
        return 2.6

# Cost function for insertion (gaps)
def insertion_cost(note, k2):
    """Calculates the insertion cost for a given note."""
    return k2 * note[1]  # k2 * Length of inserted note (duration)

# Cost function for deletion
def deletion_cost(note, k2):
    """Calculates the deletion cost for a given note."""
    return k2 * note[1]  # k2 * Length of deleted note (duration)

# Cost function for substitution (replacement)
def substitution_cost(note1, note2, k2):
    """
    Calculates the substitution cost between two notes.
    The cost involves interval differences and duration differences.
    It is defined as k2 times the sum of the lengths of the two notes minus the weight of their interval differences and duration differences.

    Args:
    - note1: A tuple containing (pitch, duration) for the first note.
    - note2: A tuple containing (pitch, duration) for the second note.

    Returns:
    - The calculated substitution cost.
    """
    # Calculate interval weight based on pitch difference
    interval_degree_weight = get_degree_weight(note1[0] - note2[0])

    # Calculate interval cost as interval weight * shorter duration
    interval_weight = interval_degree_weight * min(note1[1], note2[1])

    # Calculate duration weight as absolute difference in duration
    duration_weight = abs(note1[1] - note2[1])

    # Total substitution cost
    return k2 * (note1[1] + note2[1]) - (interval_weight + duration_weight)

# Mongeau-Sankoff Alignment Function with Local Alignment
def mongeau_sankoff_alignment(sequence1, sequence2, k2):
    """
    Perform Mongeau-Sankoff alignment between two sequences.

    Args:
    - sequence1: List of tuples, where each tuple contains (pitch, duration) for sequence 1.
    - sequence2: List of tuples, where each tuple contains (pitch, duration) for sequence 2.

    Returns:
    - alignment_cost: The calculated alignment cost, representing the quality of the best alignment.
    """

    m, n = len(sequence1), len(sequence2)
    # Step 1: Initialize alignment matrix with zeros for local alignment
    alignment_matrix = np.zeros((m + 1, n + 1))
    
    max_value = 0

    # Step 2: Fill the alignment matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Consider different segment lengths for more complex substitution scoring
            score_sub = float('-inf')

            # Step 2.1: Segment scoring loop          
            for k in range(1, min(i, j) + 1):  # Loop over possible segment lengths
                segment_score = 0
                for s in range(k):
                    segment_score += substitution_cost(sequence1[i - s - 1], sequence2[j - s - 1], k2)
                score_sub_segment = alignment_matrix[i - k][j - k] + segment_score
                score_sub = max(score_sub, score_sub_segment)


            # Step 2.2: Calculate scores for deletion and insertion
            score_del = alignment_matrix[i - 1][j] + deletion_cost(sequence1[i - 1], k2)
            score_ins = alignment_matrix[i][j - 1] + insertion_cost(sequence2[j - 1], k2)

            # Step 2.3: Select the highest score or zero (for local alignment)
            alignment_matrix[i][j] = max(0, score_sub, score_del, score_ins)

            # Track the maximum score for traceback
            if alignment_matrix[i][j] > max_value:
                max_value = alignment_matrix[i][j]



    # Alignment quality is represented by the maximum value found for local alignment
    alignment_quality = max_value

    return alignment_quality