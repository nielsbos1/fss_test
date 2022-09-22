import math
import numpy as np
import hashlib
import struct

def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = (1 << 32)


class FillSketch(object):
    def __init__(self, input, sketch_length=128, seed=1, hashfunc=sha1_hash32):
        if sketch_length > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.list_seeds = None
        self.input = input
        self.seed = seed
        self.sketch_length = sketch_length
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc

        self.hash_outputs = {x: self.get_hash_values(x) for x in self.input}

        # generate hash
        self.hashvalues = self._generate_fill_sketch(input_set=input,
                                                     sketch_length=sketch_length)


    def _generate_fill_sketch(self, input_set, sketch_length):
        sketch = np.repeat(math.inf, repeats=sketch_length)
        c = 0
        for i in range(2 * sketch_length):
            for input in input_set:
                if i < sketch_length:
                    bin_value = int(self.hash_outputs[input][i])
                    v_value = i
                else:
                    bin_value = i - sketch_length
                    v_value = i
                if math.isinf(sketch[bin_value]):
                    c += 1
                sketch[bin_value] = min(sketch[bin_value], v_value)
            if c == sketch_length:
                return sketch
        return sketch


    def get_hash_values(self, input):
        hash_value = self.hashfunc(input.encode('utf-8'))
        a, b = self._init_permutations(self.sketch_length)
        phv = np.bitwise_and((a * hash_value + b) % _mersenne_prime, self.sketch_length - 1)
        return phv


    def _init_permutations(self, num_perm):
        # Create parameters for a random bijective permutation function
        # that maps a 32-bit hash value to another 32-bit hash value.
        # http://en.wikipedia.org/wiki/Universal_hashing
        gen = np.random.RandomState(seed=self.seed)
        return np.array([
            (gen.randint(1, _mersenne_prime, dtype=np.uint64), gen.randint(0, _mersenne_prime, dtype=np.uint64)) for _ in range(num_perm)
        ], dtype=np.uint64).T

    def get_estimated_jaccard_similarity(self, other_fillsketch):
        return float(np.sum(self.hashvalues == other_fillsketch.hashvalues) / np.shape(self.hashvalues))


    def __len__(self):
        '''
        :returns: int -- The number of hash values.
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        :returns: bool -- If their seeds and hash values are both equal then two are equivalent.
        '''
        return type(self) is type(other) and \
            self.seed == other.seed and \
            np.array_equal(self.hashvalues, other.hashvalues)
