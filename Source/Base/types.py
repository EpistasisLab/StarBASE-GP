#####################################################################################################
#
# Types to be used across all files within the project. Helps with consistency and readability.
#
#####################################################################################################

import numpy as np

# float type: used for r2 scores, distances
float32_t = np.float32

# uint16 type
uint16_t = np.uint16

# uint32 type
uint32_t = np.uint32

# int8 type
int8_t = np.int8

# int16 type
int16_t = np.int16

# int32 type
int32_t = np.int32

# numpy random number generator type
rng_t = np.random.Generator

# probability type: needed to avoid rounding errors with probabilities
prob_t = np.float64

# SNP type
snp_t = np.str_