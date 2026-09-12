"""AF35-4 -- a split control envelope must re-slope when the structure moves.

An envelope applied with ``control=True`` and then split by a per-node
instrument change stores one descriptor per instrument run.  Each descriptor
carries a ``curve_window``: the slice of the WHOLE curve its own leaves cover,
which is what makes a split reproduce the values the unsplit envelope had
rather than restarting the gesture inside each half (``_curve_windows``).

Before this file, two things about that window were only ever computed once:

*   ``baked_timing`` -- the rebake gate's staleness signature -- was normalised
    to the descriptor's OWN leaves.  A one-leaf run therefore signed as the
    constant ``((0.0, 1.0),)`` and could never differ from itself, so its gate
    could not fire; a multi-leaf run signed blind to its own POSITION inside
    the whole span.
*   ``curve_window`` itself was frozen at split time, so even a descriptor that
    did rebake sampled a slice of the curve that no longer matched where its
    leaves sit.

**The expected values here are hand-computed from the envelope's geometry, not
read back from Klotho.**  The unit is four leaves of a 4/4 bar at 120 bpm, so
the span is 2.0 s, and ``Envelope([0.2, 0.8])`` is linear across it::

    value(t) = 0.2 + (0.8 - 0.2) * t / 2.0 = 0.2 + 0.3 * t

``endpoint=True`` (the default) makes the span end at the last leaf's release,
which every edit below leaves at t = 2.0 s.  So each expected row below is just
``0.2 + 0.3 * onset`` at the post-edit onsets, which are themselves a property
of ``scale`` and not of the envelope code under test.  The unsplit unit is
carried alongside as a second, independent oracle: splitting is bookkeeping, so
the two must agree.
"""
import pytest

from klotho.dynatos import Envelope
from klotho.thetos import CompositionalUnit as UC


def _build(split: bool) -> UC:
    uc = UC(tempus='4/4', prolatio=(1, 1, 1, 1), beat='1/4', bpm=120)
    uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
    uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', 0, control=True)
    if split:
        # the middle leaf gets its own timbre, which splits the overlay into
        # three runs: [leaf0], [leaf1], [leaf2, leaf3]
        uc.set_instrument(uc._rt.leaf_nodes[1], 'kl_pluck')
    return uc


def _amps(uc: UC) -> list[float]:
    return [round(uc.pt[n]['amp'], 6) for n in uc._rt.leaf_nodes]


def _onsets(uc: UC) -> list[float]:
    uc._ensure_timing_cache()
    return [uc._real_times[n]['real_onset'] for n in uc._rt.leaf_nodes]


# onsets after each edit, and 0.2 + 0.3 * onset evaluated on them by hand
EDITS = [
    ((3, 9), [0.2, 0.25, 0.3, 0.35]),       # onsets 0, 1/6, 1/3, 1/2
    ((0, 3), [0.2, 0.5, 0.6, 0.7]),         # onsets 0, 1, 4/3, 5/3
    ((1, 7), [0.2, 0.26, 0.68, 0.74]),      # onsets 0, 0.2, 1.6, 1.8
]


def test_the_split_really_happens():
    """Guard on the fixture: without three descriptors this file tests nothing."""
    assert len(_build(split=True)._control_envelopes) == 3
    assert len(_build(split=False)._control_envelopes) == 1


def test_splitting_alone_does_not_move_a_value():
    """Ryan's ruling: a split is bookkeeping, it does not restart the gesture."""
    # 0.2 + 0.3 * [0, 0.5, 1.0, 1.5]
    assert _amps(_build(split=True)) == [0.2, 0.35, 0.5, 0.65]
    assert _amps(_build(split=False)) == [0.2, 0.35, 0.5, 0.65]


@pytest.mark.parametrize("edit,expected", EDITS)
def test_split_envelope_reslopes_to_the_hand_computed_ramp(edit, expected):
    uc = _build(split=True)
    uc._rt.scale(*edit)
    assert [round(0.2 + 0.3 * o, 6) for o in _onsets(uc)] == expected, (
        "the hand calculation no longer describes this unit's geometry; "
        "fix the expectation, not the code"
    )
    assert _amps(uc) == expected


@pytest.mark.parametrize("edit,expected", EDITS)
def test_the_unsplit_control_agrees(edit, expected):
    uc = _build(split=False)
    uc._rt.scale(*edit)
    assert _amps(uc) == expected


def test_two_different_edits_give_two_different_results():
    """The frozen-signature symptom in its plainest form.

    ``((0.0, 1.0),)`` is what a one-leaf run normalises to whatever its onset
    and duration are, so the gate compared a constant against itself.
    """
    a, b = _build(split=True), _build(split=True)
    a._rt.scale(0, 3)
    b._rt.scale(1, 7)
    assert _amps(a) != _amps(b)


@pytest.mark.parametrize("edit,expected", EDITS)
def test_the_playback_window_follows_the_structure(edit, expected):
    """``resolved_control_envelopes`` feeds the control-bus automation.

    A descriptor that re-baked correctly but published a stale ``curve_window``
    would put the right numbers in ``uc.events`` and the wrong ramp in the
    audio -- the same class of silent divergence, one layer down.
    """
    uc = _build(split=True)
    uc._rt.scale(*edit)
    uc._ensure_timing_cache()
    whole_start = min(_onsets(uc)) + uc._offset
    whole_end = max(uc._real_times[n]['real_onset'] + uc._offset
                    + abs(uc._real_times[n]['real_duration'])
                    for n in uc._rt.leaf_nodes)
    total = whole_end - whole_start

    for entry in uc.resolved_control_envelopes():
        nodes = entry['target_nodes']
        start = min(uc._real_times[n]['real_onset'] + uc._offset for n in nodes)
        end = max(uc._real_times[n]['real_onset'] + uc._offset
                  + abs(uc._real_times[n]['real_duration']) for n in nodes)
        want = (round((start - whole_start) / total, 9),
                round((end - whole_start) / total, 9))
        got = tuple(round(v, 9) for v in entry['curve_window'])
        assert got == want, f"stale window on {nodes}: {got} != {want}"
