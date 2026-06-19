import pytest

from razorpay_qa.utils.config import load_settings
from razorpay_qa.utils.ingest import build_clause_index, load_source


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def index(settings):
    doc = load_source(settings.pdf_path)
    return build_clause_index(doc)
