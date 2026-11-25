"""Scaffold splitting must be leak-free -- that is the whole reason it exists."""

import pytest

from mgnn.data import load_tox21_frame
from mgnn.splits import murcko_scaffold, scaffold_groups, scaffold_split

SAMPLE = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC",      # toluene series: one scaffold
    "c1ccncc1C", "c1ccncc1CC",                      # pyridine series: another
    "C1CCCCC1", "C1CCCCC1C",                        # cyclohexane
    "CCCC", "CCCCC",                                # acyclic: empty scaffold
    "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C",            # naphthalene
]


def test_scaffold_of_analogues_is_identical():
    assert murcko_scaffold("c1ccccc1C") == murcko_scaffold("c1ccccc1CCCC")


def test_substituents_are_stripped_rings_are_kept():
    assert murcko_scaffold("c1ccccc1C(=O)N") == "c1ccccc1"


def test_groups_partition_the_input():
    groups = scaffold_groups(SAMPLE)
    covered = sorted(i for idx in groups.values() for i in idx)
    assert covered == list(range(len(SAMPLE)))


def test_split_is_a_partition():
    tr, va, te = scaffold_split(SAMPLE)
    assert sorted(tr + va + te) == list(range(len(SAMPLE)))
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_no_scaffold_appears_on_both_sides():
    tr, va, te = scaffold_split(SAMPLE)
    s_tr = {murcko_scaffold(SAMPLE[i]) for i in tr}
    s_va = {murcko_scaffold(SAMPLE[i]) for i in va}
    s_te = {murcko_scaffold(SAMPLE[i]) for i in te}
    assert not (s_tr & s_va), "scaffold leaked between train and valid"
    assert not (s_tr & s_te), "scaffold leaked between train and test"
    assert not (s_va & s_te), "scaffold leaked between valid and test"


def test_split_is_deterministic():
    assert scaffold_split(SAMPLE) == scaffold_split(SAMPLE)


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        scaffold_split(SAMPLE, 0.8, 0.3, 0.1)


def test_leak_free_on_real_tox21():
    """The property that matters, checked on the actual data set."""
    df = load_tox21_frame()
    smiles = df["smiles"].tolist()
    tr, va, te = scaffold_split(smiles, 0.8, 0.1, 0.1)
    assert len(tr) + len(va) + len(te) == len(smiles)
    s_tr = {murcko_scaffold(smiles[i]) for i in tr}
    s_te = {murcko_scaffold(smiles[i]) for i in te}
    s_va = {murcko_scaffold(smiles[i]) for i in va}
    assert not (s_tr & s_te)
    assert not (s_tr & s_va)
    assert not (s_va & s_te)
    # train should hold roughly its quota; groups are never cut so it is approximate
    assert 0.70 < len(tr) / len(smiles) < 0.85
