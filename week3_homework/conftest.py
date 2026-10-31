from patch_agent import patch_agent, reset_usage, unpatch_agent, usage_summary


def pytest_sessionstart(session):
    reset_usage()
    patch_agent()


def pytest_terminal_summary(terminalreporter):
    terminalreporter.section("LLM usage and estimated cost")
    terminalreporter.write_line(usage_summary())


def pytest_unconfigure(config):
    unpatch_agent()
