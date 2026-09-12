"""AF35-3 -- the two public doors to `set_instrument` must answer the same way.

An overlay splits at an instrument change (Ryan, 2026-08-31). ``UC.set_instrument``
has always done that eagerly at the end of its own write. The tree door,
``uc._rt.set_instrument``, did not -- so the same operation through the other
handle left the unit in a different state (one control-envelope descriptor
against three), and the split was then attempted LAZILY from inside
``RhythmTree._respell``, where the new nodes have no metric layer yet::

    TREE door:  scale -> KeyError: 'metric_duration'   insert -> KeyError
                subdivide -> ok                        replace_node -> ok
    UC   door:  scale -> ok                            insert -> ok

Two public doors to one operation answering differently is R12's
programmer's-lens failure in its plainest form, and the docstring twelve lines
above the offending call already named this exact hazard.

The invariant that outlives the two specific verbs is
:func:`test_no_descriptor_straddles_an_instrument_boundary`: the split exists to
keep control messages inside one instrument, so no descriptor may ever span two.
That is what would catch a heal LOST to the gating, as opposed to a crash.
"""
import pytest

from klotho.dynatos import Envelope
from klotho.thetos import CompositionalUnit as UC

DOORS = ('TREE', 'UC')
VERBS = ('scale', 'insert', 'subdivide', 'replace_node')


def _build(door):
    uc = UC(tempus='4/4', prolatio=(1, 1, 1, 1), beat='1/4', bpm=120)
    uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
    uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', 0, control=True)
    leaf = uc._rt.leaf_nodes[1]
    if door == 'TREE':
        uc._rt.set_instrument(leaf, 'kl_pluck')
    else:
        uc.set_instrument(leaf, 'kl_pluck')
    return uc


def _edit(uc, verb):
    leaf = uc._rt.leaf_nodes[1]
    if verb == 'scale':
        uc._rt.scale(0, 3)
    elif verb == 'insert':
        uc._rt.insert(0, '1/4')
    elif verb == 'subdivide':
        uc._rt.subdivide(leaf, (1, 1))
    elif verb == 'replace_node':
        uc._rt.replace_node(leaf, proportion=3)


def _windows(uc):
    return sorted(tuple(round(v, 9) for v in d.get('curve_window') or (0.0, 1.0))
                  for d in uc._control_envelopes.values())


def _amps(uc):
    # ``insert`` adds a leaf OUTSIDE the envelope's span, which correctly has
    # no value for the controlled pfield. ``None`` is the honest reading and
    # is compared as such; collapsing it to a number would hide a real
    # difference between the doors.
    return [None if uc.pt[n]['amp'] is None else round(uc.pt[n]['amp'], 6)
            for n in uc._rt.leaf_nodes]


def test_the_tree_door_splits_the_overlay_at_once():
    """State parity, with no structural edit involved at all.

    This is the half a crash-only test would miss: before the fix the tree door
    left ONE descriptor spanning the instrument change, which is the state the
    ruling forbids, and it stayed that way until the next structural edit.
    """
    assert len(_build('TREE')._control_envelopes) == 3
    assert _windows(_build('TREE')) == _windows(_build('UC'))


@pytest.mark.parametrize("verb", VERBS)
def test_neither_door_raises(verb):
    for door in DOORS:
        _edit(_build(door), verb)   # the KeyError was here


@pytest.mark.parametrize("verb", VERBS)
def test_the_two_doors_agree_after_the_same_edit(verb):
    results = []
    for door in DOORS:
        uc = _build(door)
        _edit(uc, verb)
        results.append((len(uc._control_envelopes), _windows(uc), _amps(uc)))
    assert results[0] == results[1]


@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("door", DOORS)
def test_no_descriptor_straddles_an_instrument_boundary(door, verb):
    """The invariant the split exists for, checked after the edit.

    Gating the lazy split on the leaf-surface announcement moves WHEN it runs.
    If that ever dropped a heal instead of deferring it, a descriptor would be
    left spanning two instruments and this is what would say so.
    """
    uc = _build(door)
    _edit(uc, verb)
    for desc in uc._control_envelopes.values():
        leaves = uc._resolve_control_envelope_leaves(desc)
        assert len(uc._partition_by_instrument(leaves)) == 1, (
            f"descriptor spans an instrument change: {leaves}")


def test_scale_gives_the_hand_computed_ramp_through_both_doors():
    """0.2 + 0.3 * onset, the same arithmetic oracle AF35-4's guard uses.

    Parity alone would be satisfied by both doors being wrong together.
    """
    for door in DOORS:
        uc = _build(door)
        uc._rt.scale(0, 3)
        uc._ensure_timing_cache()
        onsets = [uc._real_times[n]['real_onset'] for n in uc._rt.leaf_nodes]
        assert onsets == [0.0, 1.0, 4 / 3, 5 / 3]
        assert _amps(uc) == [0.2, 0.5, 0.6, 0.7]


def test_a_multi_node_binding_splits_against_the_finished_selection():
    """The per-node announcement is held across a whole ``UC.set_instrument``.

    Without the hold, the tree door would announce inside the loop and split
    the overlay against a half-bound selection -- three descriptors after the
    first node, then a resplit of a state that never existed as music.
    """
    uc = UC(tempus='4/4', prolatio=(1, 1, 1, 1), beat='1/4', bpm=120)
    uc.set_instrument(uc._rt.leaf_nodes, 'kl_saw')
    uc.apply_envelope(Envelope([0.2, 0.8]), 'amp', 0, control=True)
    uc.set_instrument(uc._rt.leaf_nodes[1:3], 'kl_pluck')
    # runs are [leaf0], [leaf1, leaf2], [leaf3] -- three, not four
    assert len(uc._control_envelopes) == 3
    assert _windows(uc) == [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]
