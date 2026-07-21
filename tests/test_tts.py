from unittest.mock import MagicMock, patch

from src.tts import speak


def test_speak_uses_correct_voice_for_character():
    with (
        patch("src.tts.kokoro_pipeline") as mock_pipeline,
        patch("src.tts.sf.write"),
        patch("src.tts.add_cave_echo") as mock_echo,
    ):
        mock_pipeline.return_value = [(None, None, MagicMock())]
        speak("hello", "Genie", filename="test.wav")

    called_voice = mock_pipeline.call_args.kwargs.get("voice")
    assert called_voice == "am_adam"
    mock_echo.assert_not_called()


def test_speak_applies_cave_echo_only_for_cave_of_wonders():
    with (
        patch("src.tts.kokoro_pipeline") as mock_pipeline,
        patch("src.tts.sf.write"),
        patch("src.tts.add_cave_echo") as mock_echo,
    ):
        mock_pipeline.return_value = [(None, None, MagicMock())]
        speak("hello", "The Cave of Wonders", filename="test.wav")

    mock_echo.assert_called_once_with("test.wav")
