"""Shared fixtures. ``fake_db`` uses made-up coefficients (tests/fixtures) for logic tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyrothermo import Species, load_species

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_YAML = FIXTURES / "fake_species.yaml"


@pytest.fixture(scope="session")
def fake_db() -> dict[str, Species]:
    """Species built from FAKE coefficients with closed-form properties."""
    return load_species(FAKE_YAML)
