r"""JAX kernels for the (r1, s, mu) pair histogram.

The estimator's inner loop is embarrassingly parallel and arithmetic-light -- one distance, one dot
product, three bin lookups and a scatter-add per pair -- so it is limited by memory traffic in
numpy, which materialises several (block x n2 x 3) temporaries per chunk. Under XLA the whole chunk
fuses into a single pass with no intermediates, and the same code runs on a GPU unchanged.

Two entry points, matching the two ways pairs are produced:

    far_block   all pairs between a block of primaries and every secondary, selected by a mask
                (jit needs static shapes, so out-of-range pairs are given zero weight rather than
                being filtered out);
    flat_pairs  an explicit list of pairs, as returned by the KD-tree neighbour search, processed
                in fixed-size padded chunks so that only one kernel is ever compiled.

Both accumulate the same five quantities as the numpy path: sum of the pair weights, of w*r1, w*s,
w*mu, and the unweighted count. float64 is enabled on import: the tripolar sums cancel heavily and
float32 accumulation is visible at the 1e-6 level.
"""
from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def is_uniform(edges) -> bool:
    d = jnp.diff(jnp.asarray(edges))
    return bool(jnp.all(jnp.abs(d - d[0]) <= 1e-9 * jnp.abs(d[0])))


def _index_axis(x, edges, n, uniform):
    """Bin index without searchsorted when the edges are equally spaced (they usually are)."""
    if uniform:
        return jnp.floor((x - edges[0]) / ((edges[-1] - edges[0]) / n)).astype(jnp.int32)
    return (jnp.searchsorted(edges, x, side='right') - 1).astype(jnp.int32)


def _bin_index(r1, s, mu, r_edges, s_edges, mu_edges, shape, uni):
    na, nb, nm = shape
    ia = jnp.clip(_index_axis(r1, r_edges, na, uni[0]), 0, na - 1)
    ib = _index_axis(s, s_edges, nb, uni[1])
    im = jnp.clip(_index_axis(mu, mu_edges, nm, uni[2]), 0, nm - 1)
    inside = (ib >= 0) & (ib < nb)
    ib = jnp.clip(ib, 0, nb - 1)
    return (ia * nb + ib) * nm + im, inside


def _scatter(idx, w, r1, s, mu, keep, n_cells):
    """The five accumulations, as segment sums over the flattened cell index.

    The pairwise arithmetic may run in float32, but the sums are accumulated in float64: ~1e8
    additions into ~1e5 cells would lose several digits otherwise.
    """
    w, r1, s, mu = (jnp.asarray(v, dtype=jnp.float64) for v in (w, r1, s, mu))
    ww = jnp.where(keep, w, 0.0)
    cnt = jnp.where(keep, 1.0, 0.0)
    seg = partial(jax.ops.segment_sum, segment_ids=idx, num_segments=n_cells,
                  indices_are_sorted=False, unique_indices=False)
    return (seg(ww), seg(ww * r1), seg(ww * s), seg(ww * mu), seg(cnt))


@partial(jax.jit, static_argnums=(8, 9, 12))
def far_block(p1, w1, r1, p2, w2, r_edges, s_edges, mu_edges, shape, n_cells, s_lo, s_hi, uni):
    """All pairs between the primaries p1 (a block) and every secondary p2, with s in [s_lo, s_hi)."""
    d = p2[None, :, :] - p1[:, None, :]
    s = jnp.sqrt(jnp.sum(d * d, axis=-1))
    safe = jnp.where(s > 0, s, 1.0)
    r1b = jnp.broadcast_to(r1[:, None], s.shape)
    mu = jnp.sum(p1[:, None, :] * d, axis=-1) / (jnp.where(r1b > 0, r1b, 1.0) * safe)
    mu = jnp.clip(mu, -1.0, 1.0)
    w = w1[:, None] * w2[None, :]
    keep = (s >= s_lo) & (s < s_hi) & (s > 0) & (r1b > 0)
    idx, inside = _bin_index(r1b.ravel(), s.ravel(), mu.ravel(), r_edges, s_edges, mu_edges, shape, uni)
    return _scatter(idx, w.ravel(), r1b.ravel(), s.ravel(), mu.ravel(),
                    (keep & inside.reshape(s.shape)).ravel(), n_cells)


@partial(jax.jit, static_argnums=(7, 8, 10))
def flat_pairs(x1, r1, d, w, r_edges, s_edges, mu_edges, shape, n_cells, valid, uni):
    """An explicit list of pairs: x1 the primary positions, d the separation vectors."""
    s = jnp.sqrt(jnp.sum(d * d, axis=-1))
    safe = jnp.where(s > 0, s, 1.0)
    mu = jnp.clip(jnp.sum(x1 * d, axis=-1) / (jnp.where(r1 > 0, r1, 1.0) * safe), -1.0, 1.0)
    idx, inside = _bin_index(r1, s, mu, r_edges, s_edges, mu_edges, shape, uni)
    keep = valid & inside & (s > 0) & (r1 > 0)
    return _scatter(idx, w, r1, s, mu, keep, n_cells)


def available() -> bool:
    return True
