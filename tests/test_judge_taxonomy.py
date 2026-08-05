"""The folding rule, tested without loading a model.

`classify` turns four independent Yes/No probes into one label. The probes are
what a model produces; this is the part a person decides, so it is the part
worth pinning down in a test.

The case that matters is `right_answer_denied_source`: the needle is present
AND the answer refuses to treat it as user-stated. Counting correct answers
scores that as a success, counting failures scores it as a miss, and it is
neither -- which is the whole reason the taxonomy exists.
"""

from dataclasses import dataclass

from oldnews.evals.judge import ABSTAIN, PROBES, classify


@dataclass
class V:
    yes: bool
    margin: float
    p_yes: float = 0.5


def probes(**kw):
    """All probes confidently No, except the ones named."""
    out = {k: V(False, -3.0) for k in PROBES}
    for k, v in kw.items():
        out[k] = V(bool(v), 3.0 if v else -3.0)
    return out


def test_right_answer_while_denying_the_source_is_its_own_category():
    label = classify(True, probes(states_a_value=True, denies_being_told=True))
    assert label == "right_answer_denied_source"


def test_needle_present_and_nothing_odd_is_just_recall():
    assert classify(True, probes(states_a_value=True)) == "recalled"


def test_specific_but_wrong_value_is_confabulation():
    assert classify(False, probes(states_a_value=True)) == "confabulation"


def test_declining_without_the_needle_is_a_disclaimed_non_answer():
    assert classify(False, probes(denies_being_told=True)) == "disclaimed_non_answer"


def test_broken_text_wins_over_everything_else():
    # degeneracy is checked first: a looping answer that happens to contain the
    # needle is not a recall, it is broken output
    assert classify(True, probes(degenerate=True, states_a_value=True)) == "degenerate"


def test_self_contradiction_without_the_needle():
    assert classify(False, probes(contradicts_itself=True,
                                  states_a_value=True)) == "self_contradiction"


def test_saying_nothing_at_all():
    assert classify(False, probes()) == "no_answer"


def test_one_unsure_probe_makes_the_whole_case_unsure():
    p = probes(states_a_value=True)
    p["degenerate"] = V(False, ABSTAIN / 2)
    assert classify(False, p) == "unsure"


def test_abstain_threshold_is_respected_at_the_boundary():
    p = probes(states_a_value=True)
    p["degenerate"] = V(False, -(ABSTAIN + 0.01))
    assert classify(False, p) != "unsure"
