from coco_agents import terminal


def test_terminal_colors_are_disabled_when_no_color_is_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert terminal.format_user_prompt() == ">>> "
    assert terminal.format_agent_response("answer") == "answer"


def test_terminal_colors_format_user_and_agent_text(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(terminal.sys.stdout, "isatty", lambda: True)
    assert terminal.format_user_prompt() == "\033[37m>>> \033[0m"
    assert terminal.format_agent_response("answer") == "\033[1;34manswer\033[0m"
