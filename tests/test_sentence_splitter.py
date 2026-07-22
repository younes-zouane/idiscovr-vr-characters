from src.sentence_splitter import split_into_sentences


def test_empty_buffer_returns_nothing():
    assert split_into_sentences("") == ([], "")


def test_no_boundary_yet_returns_everything_as_remainder():
    sentences, remainder = split_into_sentences("Hello there, how")
    assert sentences == []
    assert remainder == "Hello there, how"


def test_single_complete_sentence():
    sentences, remainder = split_into_sentences("This is a full sentence. And more")
    assert sentences == ["This is a full sentence."]
    assert remainder == "And more"


def test_abbreviation_does_not_split_early():
    sentences, remainder = split_into_sentences("Hello Dr. Smith. How are you? I'm great")
    assert sentences == ["Hello Dr. Smith."]
    assert remainder == "How are you? I'm great"


def test_numbered_list_does_not_split_early():
    sentences, remainder = split_into_sentences("Steps: 1. Open the lamp. Then rub it.")
    assert sentences == ["Steps: 1. Open the lamp."]
    assert remainder == "Then rub it."


def test_multiple_sentences_at_once():
    sentences, remainder = split_into_sentences("First one here. Second one too! Third")
    assert sentences == ["First one here.", "Second one too!"]
    assert remainder == "Third"
