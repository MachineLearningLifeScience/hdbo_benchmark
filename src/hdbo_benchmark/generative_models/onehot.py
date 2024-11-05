import numpy as np


class OneHot:
    def __init__(self, alphabet_s_to_i: dict[str, int], max_sequence_length: int):
        self.alphabet_s_to_i = alphabet_s_to_i
        self.alphabet_i_to_s = {v: k for k, v in alphabet_s_to_i.items()}
        self.max_sequence_length = max_sequence_length
        self.n_classes = len(alphabet_s_to_i)
        self.latent_dim = max_sequence_length * self.n_classes

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        """
        Encodes a string array into a one-hot encoded array.
        """
        one_hot_representation = np.zeros(
            (x.shape[0], self.max_sequence_length, self.n_classes)
        )

        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                one_hot_representation[i, j, self.alphabet_s_to_i[x_ij]] = 1

        one_hot_representation = one_hot_representation.reshape(-1, self.latent_dim)

        return one_hot_representation

    def decode_to_string_array(self, z: np.ndarray) -> np.ndarray:
        """
        Decodes a one-hot encoded array into a string array.
        """
        z = z.reshape(-1, self.max_sequence_length, self.n_classes)
        string_representation = np.array(
            [[self.alphabet_i_to_s[z_ij] for z_ij in z_i] for z_i in z.argmax(axis=-1)]
        )

        return string_representation
