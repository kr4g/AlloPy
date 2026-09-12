"""Four defects introduced by the AF35-4 / AF35-3 fix, plus the cost of the gate.

`split_group` links the control-envelope descriptors that came from one
``apply_envelope`` call, so a run can find the rest of its span. An adversarial
review of 4e7ec39 and 78b67d2 found four ways that linkage is wrong and one way
it is expensive. Every oracle below is either a property of the SPEC (an explicit
later write wins; two unrelated envelopes do not interact) or the unsplit /
uncopied unit computing the same music by a different path -- never a value read
back out of the code under test.
"""
import time

import pytest

from klotho.dynatos import Envelope
from klotho.thetos import CompositionalUnit as UC


def _amps(uc, key='amp'):
    return [None if uc.pt[n][key] is None else round(uc.pt[n][key], 6)
            for n in uc._rt.leaf_nodes]


def _groups(uc):
    return [d.get('split_group') for d in uc._control_envelopes.values()]


# ---------------------------------------------------------------- D1
def _split_amp_unit():
    """Six leaves, an `amp` envelope over the first three, split by a timbre."""
    uc = UC(tempus='4/4', prolatio=(1, 1, 1, 1, 1, 1), beat='1/4', bpm=120)
    uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
    uc.set_instrument(uc._rt.leaf_nodes[1], 'kl_pluck')
    uc.apply_envelope(Envelope([0.2, 0.6]), 'amp',
                      list(uc._rt.leaf_nodes[:3]), control=True)
    return uc


def _add_pan_envelope(uc):
    """A SECOND, unrelated envelope over the other three leaves."""
    L = uc._rt.leaf_nodes
    uc.set_instrument(L[4], 'kl_pluck')
    uc.apply_envelope(Envelope([0.0, 1.0]), 'pan', list(L[3:]), control=True)
    return uc


def test_a_copy_does_not_mint_a_group_id_it_already_carries():
    """Three copy paths carried `split_group` but not the counter that mints it.

    `uc * Fraction(k)`, `modulate_tempo` and `modulate_tempus` each copied the
    descriptors and `_next_envelope_id` and stopped, leaving
    `_next_envelope_split_group` at 0 while the descriptors they had just copied
    carried group 0. The next envelope authored on that unit minted group 0
    again, so two unrelated envelopes became one group: every run was then
    normalised against the union's span and re-sloped against a partition that
    describes neither.

    The oracle is the same pan envelope with no amp envelope anywhere near it.
    """
    from fractions import Fraction

    merged = _add_pan_envelope(_split_amp_unit() * Fraction(1, 1))
    groups = [g for g in _groups(merged) if g is not None]
    assert len(set(groups)) == 2, (
        f"an unrelated envelope joined the copied envelope's group: {groups}")

    # oracle: the pan envelope alone, reached by the identical route
    bare = UC(tempus='4/4', prolatio=(1, 1, 1, 1, 1, 1), beat='1/4', bpm=120)
    bare.set_instrument(bare._rt.leaf_nodes, 'kl_saw')
    bare.set_instrument(bare._rt.leaf_nodes[1], 'kl_pluck')
    oracle = _add_pan_envelope(bare * Fraction(1, 1))

    merged._rt.scale(0, 3)
    oracle._rt.scale(0, 3)
    assert _amps(merged, 'pan') == _amps(oracle, 'pan'), (
        "an unrelated amp envelope changed what the pan envelope does")


# ---------------------------------------------------------------- D2
def test_an_edit_outside_the_span_does_not_revert_a_later_explicit_write():
    """ENV-6 promises last-write-wins; a re-split must not break it.

    `_split_envelope_at_instrument_changes` recorded each new descriptor's
    staleness signature against the leaves of the descriptor it was cutting,
    while every reader normalises against the whole GROUP. The two bases differ
    the moment the descriptor being cut is itself a group member, so the new
    descriptors read stale on creation with nothing moved -- and the next edit
    ANYWHERE rebaked them over the composer's own later write.
    """
    uc = UC(tempus='4/4', prolatio=(1, 1, 1, 1, 1, 1), beat='1/4', bpm=120)
    uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
    leaves = uc._rt.leaf_nodes
    uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', list(leaves[:4]), control=True)
    uc.set_instrument(leaves[3], 'kl_pluck')     # first cut
    uc.set_instrument(leaves[1], 'kl_pluck')     # re-split of a group member
    uc._rt.set_pfields(leaves[0], amp=0.9)       # the composer's later word
    assert _amps(uc)[0] == 0.9

    uc._rt.subdivide(leaves[5], (1, 1))          # strictly OUTSIDE the span
    assert _amps(uc)[0] == 0.9, "an edit outside the span reverted an explicit write"


# ---------------------------------------------------------------- D3
@pytest.mark.parametrize("endpoint", [True, False])
def test_a_split_tracks_the_unsplit_unit_for_either_endpoint(endpoint):
    """`resolved_control_envelopes` feeds the audio; `uc.pt` feeds `uc.events`.

    The staleness signature always ended the span at ``max(onset + duration)``
    while `_curve_windows` ends it at ``max(onset)`` when ``endpoint`` is false.
    A change to the last onset therefore moved the windows and left the signature
    identical, so a sibling's rebake refreshed this member's PUBLISHED window
    while the gate declined to rebake its VALUES. Right numbers on screen, wrong
    ramp in the audio -- AF35-43 one layer down.
    """
    def build(split):
        uc = UC(tempus='4/4', prolatio=(1, 1, 1, 1), beat='1/4', bpm=120)
        uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
        uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', 0,
                          control=True, endpoint=endpoint)
        if split:
            uc.set_instrument(uc._rt.leaf_nodes[1], 'kl_pluck')
        uc._rt.subdivide(uc._rt.leaf_nodes[-1], (1, 1))
        return uc

    # The unsplit unit is the oracle: splitting is bookkeeping, so the two must
    # agree whatever `endpoint` is. Asserting the window's arithmetic directly
    # cannot be done here -- a one-leaf run's window is legitimately degenerate
    # (zero width) and `_bake_envelope` keeps its position while borrowing the
    # width, which is documented. The values are the thing that must match.
    assert _amps(build(True)) == _amps(build(False))


# ---------------------------------------------------------------- D4
def test_one_rested_run_does_not_refreeze_the_whole_group():
    """The group helper bailed if ANY member resolved to no sounding leaves.

    Resting the only leaf of one instrument run made that run resolve to `[]`,
    so every other member -- whose geometry is perfectly readable -- fell back to
    self-normalised signatures and froze again. Oracle: the unsplit envelope over
    the same music, which re-slopes.
    """
    def build(split):
        uc = UC(tempus='6/4', prolatio=(1,) * 6, beat='1/4', bpm=120)
        uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
        if split:
            # runs [L0] [L1] [L2..L5] -- the middle run is a single leaf, so
            # resting it is what empties a run without destroying the envelope
            uc.set_instrument(uc._rt.leaf_nodes[1], 'kl_pluck')
        uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', 0, control=True)
        # the TREE door: uc.make_rest() deletes the descriptor outright, which
        # is a different case and not the one that froze the group
        uc._rt.make_rest(uc._rt.leaf_nodes[1])
        return uc

    split, control = build(True), build(False)
    before = _amps(split)
    split._rt.scale(0, 3)
    control._rt.scale(0, 3)
    assert _amps(split) != before, "the split group froze when one run was rested"
    assert _amps(split) == _amps(control)


# ---------------------------------------------------------------- D5
def test_the_rebake_gate_is_linear_in_the_number_of_descriptors():
    """The gate resolved every member's leaves once PER DESCRIPTOR.

    Counted rather than timed: a wall-clock bound is flaky, and the defect is a
    complexity class, not a constant. With n descriptors one announcement did
    O(n^2) resolves; doubling n must not quadruple the count.
    """
    def resolves_for(n_leaves):
        uc = UC(tempus='4/4', prolatio=tuple([1] * n_leaves), beat='1/4', bpm=120)
        uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
        uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', 0, control=True)
        for i, n in enumerate(uc._rt.leaf_nodes):
            if i % 2:
                uc._rt.set_instrument(n, 'kl_pluck')
        calls = [0]
        real = type(uc)._resolve_control_envelope_leaves

        def counted(self, desc):
            calls[0] += 1
            return real(self, desc)

        type(uc)._resolve_control_envelope_leaves = counted
        try:
            uc._rt.scale(0, 3)
        finally:
            type(uc)._resolve_control_envelope_leaves = real
        return calls[0], len(uc._control_envelopes)

    c32, d32 = resolves_for(32)
    c64, d64 = resolves_for(64)
    assert d64 > d32 >= 16
    # quadratic would be ~4x; linear is ~2x. 3x is a generous ceiling that still
    # separates the two classes by a wide margin.
    assert c64 < c32 * 3, (
        f"resolve count grew {c64 / c32:.1f}x for 2x the descriptors "
        f"({c32} at {d32} descriptors, {c64} at {d64}) -- still superlinear")
