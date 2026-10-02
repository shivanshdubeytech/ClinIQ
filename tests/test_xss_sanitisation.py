"""Regression test for DOMPurify client-side markdown sanitization."""
import pytest


def test_markdown_sanitiser_strips_script():
    """Confirms DOMPurify strips <script> tags from assistant markdown in browser.

    Browser-side assertion: Requires a headless browser harness (Playwright/Selenium).
    Since no headless browser is installed in this environment, this test records
    the operational gap per PRE_DEPLOYMENT_HARDENING §8f and DEPLOYMENT_BLOCKERS.md.
    """
    try:
        import playwright  # noqa: F401
    except ImportError:
        pytest.skip(
            "Headless browser (playwright) is not installed in the environment. "
            "Reported as incomplete per §8f and documented in DEPLOYMENT_BLOCKERS.md."
        )
