"""The probe batches' neighbour matrix (jax_clashes._neighbour_matrix) against the row-by-row fill it replaced"""
import numpy as np

from remolprobity import jax_clashes


def _row_by_row(dir_nb, b_starts, b_counts, b_pad, k_pad):
    nb_mat = np.zeros((b_pad, k_pad), dtype=np.int32)
    nb_val = np.zeros((b_pad, k_pad), dtype=bool)
    for r_i in range(len(b_counts)):
        c_i, s_i = b_counts[r_i], b_starts[r_i]
        nb_mat[r_i, :c_i] = dir_nb[s_i:s_i + c_i]
        nb_val[r_i, :c_i] = True
    return nb_mat, nb_val


def test_neighbour_matrix_is_the_row_by_row_fill():
    rng = np.random.default_rng(0)
    for n_src, k_pad, b_pad in ((1, 32, 256), (37, 32, 256), (2048, 64, 2048), (300, 128, 512)):
        counts = rng.integers(1, k_pad + 1, size=n_src * 2)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        dir_nb = rng.integers(0, 10_000, size=int(counts.sum())).astype(np.int64)
        # a batch is a run of the sources, not necessarily from the first
        pick = slice(n_src // 2, n_src // 2 + n_src)
        got = jax_clashes._neighbour_matrix(dir_nb, starts[pick], counts[pick], b_pad, k_pad)
        want = _row_by_row(dir_nb, starts[pick], counts[pick], b_pad, k_pad)
        for g, w in zip(got, want):
            assert g.dtype == w.dtype and g.shape == w.shape
            assert np.array_equal(g, w)
