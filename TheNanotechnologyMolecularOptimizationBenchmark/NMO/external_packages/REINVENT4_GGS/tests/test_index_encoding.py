"""Tests for index tokenizer, vocabulary, and Action_Space roundtrip."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reinvent.models.reinvent.models.vocabulary import (
    IndexTokenizer,
    create_index_vocabulary,
)
from reinvent.action_space import Action_Space_GroupSelfies


GRAMMAR_PATH = "./GS_complex_grammar_without_S.txt"   # adjust to your actual grammar file


def test_tokenizer_roundtrip():
    tok = IndexTokenizer()
    original = "3 0 7 2 1"
    tokens = tok.tokenize(original, with_begin_and_end=True)
    assert tokens[0] == "^"
    assert tokens[-1] == "$"
    assert tok.untokenize(tokens) == original


def test_tokenizer_no_begin_end():
    tok = IndexTokenizer()
    assert IndexTokenizer().tokenize("0 1 2", with_begin_and_end=False) == ["0", "1", "2"]


def test_vocabulary_size():
    action_space = Action_Space_GroupSelfies.from_grammar_path(GRAMMAR_PATH)
    vocab = create_index_vocabulary(action_space.N_actions, IndexTokenizer())
    assert len(vocab) == action_space.N_actions + 2   # +2 for '$' and '^'


def test_vocabulary_encode_decode():
    action_space = Action_Space_GroupSelfies.from_grammar_path(GRAMMAR_PATH)
    tok = IndexTokenizer()
    vocab = create_index_vocabulary(action_space.N_actions, tok)
    tokens = tok.tokenize("3 0 7", with_begin_and_end=False)
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_all_indices_in_vocabulary():
    action_space = Action_Space_GroupSelfies.from_grammar_path(GRAMMAR_PATH)
    vocab = create_index_vocabulary(action_space.N_actions, IndexTokenizer())
    for i in range(action_space.N_actions):
        assert str(i) in vocab


@pytest.fixture(scope="module")
def action_space():
    return Action_Space_GroupSelfies.from_grammar_path(GRAMMAR_PATH)


def test_encoding_roundtrip(action_space):
    """encoding_to_action_sequence -> action_sequence_to_encoding must be stable."""
    sample_group = list(action_space.selfies_grammar.vocab.keys())[0]
    sample_gsf   = f"[:0{sample_group}]"

    action_seq   = action_space.encoding_to_action_sequence(sample_gsf)
    assert len(action_seq) > 0

    recovered    = action_space.action_sequence_to_encoding(action_seq)
    assert len(recovered) > 0


def test_has_end_token(action_space):
    end_idx = action_space.reversed_action_space["End"]
    assert action_space.has_end_token([0, 1, 2, end_idx]) is True
    assert action_space.has_end_token([0, 1, end_idx, 2]) is False  # action after End
    assert action_space.has_end_token([0, 1, 2])           is False  # no End token


def test_index_string_roundtrip(action_space):
    """Simulate the exact path the scoring plugin takes."""
    sample_group = list(action_space.selfies_grammar.vocab.keys())[0]
    action_seq   = action_space.encoding_to_action_sequence(f"[:0{sample_group}]")
    index_str    = " ".join(str(a) for a in action_seq)
    recovered    = [int(x) for x in index_str.strip().split()]
    assert recovered == action_seq