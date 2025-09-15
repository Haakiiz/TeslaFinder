import unittest

from LLM_Summariser import parse_selected_ids, shortlist_by_heuristics


class TestSummariserHelpers(unittest.TestCase):
    def test_parse_selected_ids_extracts_ids_and_urls(self):
        payload = (
            "- ad_id: 123456789 | [URL](https://www.finn.no/mobility/item/123456789)\n"
            "Also consider https://www.finn.no/mobility/item/987654321"
        )
        ids, urls = parse_selected_ids(payload)
        self.assertIn("123456789", ids)
        self.assertIn("987654321", ids)
        self.assertIn("https://www.finn.no/mobility/item/123456789", urls)

    def test_shortlist_by_heuristics_prefers_price_year_then_mileage(self):
        listings = [
            {"ad_id": "1", "price": 310000, "year": 2022, "mileage": 32000},
            {"ad_id": "2", "price": 300000, "year": 2021, "mileage": 28000},
            {"ad_id": "3", "price": 300000, "year": 2023, "mileage": 40000},
        ]
        result = shortlist_by_heuristics(listings, limit=2)
        self.assertEqual([item["ad_id"] for item in result], ["3", "2"])


if __name__ == "__main__":
    unittest.main()

