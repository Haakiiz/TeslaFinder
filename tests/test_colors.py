import pytest
import sys
from pathlib import Path

# Ensure repository root is on sys.path when tests run under pytest's importlib mode
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scrape import Listing, matches_filters


SPEC_BASE = {
    "price_max": 1_000_000,
    "year_min": 2020,
    "mileage_max": 100_000,
    "exclude_keywords": [],
}


def make_spec(**kwargs):
    spec = SPEC_BASE.copy()
    spec.update(kwargs)
    return spec


@pytest.mark.parametrize("nocolor", ["svart", "sort"])
def test_norwegian_black_matches_spec(nocolor):
    spec = make_spec(color=["black"])
    lst = Listing(ad_id="1", url="", price=0, year=2022, mileage=0, color=nocolor, location="")
    assert matches_filters(lst, spec)


def test_norwegian_black_rejected_when_not_in_spec():
    spec = make_spec(color=["white"])
    lst = Listing(ad_id="1", url="", price=0, year=2022, mileage=0, color="svart", location="")
    assert not matches_filters(lst, spec)
