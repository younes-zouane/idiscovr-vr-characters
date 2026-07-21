from unittest.mock import MagicMock, patch

from src.llm import MAX_HISTORY_MESSAGES, ask_character, init_conversation_histories


def _fake_openai_response(text="a reply"):
    fake_choice = MagicMock()
    fake_choice.message.content = text
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


def test_init_conversation_histories_has_one_entry_per_character():
    histories = init_conversation_histories()
    from src.characters import CHARACTERS

    assert set(histories.keys()) == set(CHARACTERS.keys())


def test_init_conversation_histories_starts_with_system_prompt_only():
    histories = init_conversation_histories()
    for name, history in histories.items():
        assert len(history) == 1
        assert history[0]["role"] == "system"


def test_ask_character_appends_user_and_assistant_messages():
    history = [{"role": "system", "content": "You are a test character."}]
    with patch("src.llm.client") as mock_client:
        mock_client.chat.completions.create.return_value = _fake_openai_response("Hello there!")
        reply = ask_character("Genie", "hi", history)

    assert reply == "Hello there!"
    assert history[-2] == {"role": "user", "content": "hi"}
    assert history[-1] == {"role": "assistant", "content": "Hello there!"}


def test_ask_character_caps_history_length():
    # start with a history already past the cap
    history = [{"role": "system", "content": "sys"}]
    history += [{"role": "user", "content": f"msg {i}"} for i in range(30)]

    with patch("src.llm.client") as mock_client:
        mock_client.chat.completions.create.return_value = _fake_openai_response("ok")
        ask_character("Genie", "one more", history)

    assert len(history) <= MAX_HISTORY_MESSAGES
    # system prompt must survive the trim, always at index 0
    assert history[0] == {"role": "system", "content": "sys"}


def test_two_characters_have_independent_histories():
    histories = init_conversation_histories()
    with patch("src.llm.client") as mock_client:
        mock_client.chat.completions.create.return_value = _fake_openai_response("reply A")
        ask_character("Genie", "hello genie", histories["Genie"])

    # Iago's history must be untouched by Genie's conversation
    assert len(histories["Iago"]) == 1
    assert len(histories["Genie"]) == 3  # system + user + assistant
