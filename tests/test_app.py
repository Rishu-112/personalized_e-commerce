"""Headless smoke test of every Streamlit page (skipped if demo artifacts are not trained)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "demo" / "bundle.joblib").exists(),
                                reason="train the demo model first: python -m src.train --dataset demo")

PAGES = ["Overview", "Personalized recommendations", "Similar products", "Customer segmentation", "Model performance", "Cold-start"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
