import json
import unittest

from LLM_Summariser import (
    extract_description_from_html,
    parse_selected_ids,
    shortlist_by_heuristics,
)


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


    def test_extract_description_prefers_next_data_over_meta(self):
        full_text = "Velholdt Tesla Model Y med varmepumpe. " * 20
        next_data = {"props": {"pageProps": {"ad": {"description": full_text}}}}
        html = f"""
        <html><head>
          <meta name="description" content="Velholdt Tesla Model Y med...">
        </head><body>
          <script id="__NEXT_DATA__" type="application/json">{json.dumps(next_data)}</script>
        </body></html>
        """
        result = extract_description_from_html(html)
        self.assertEqual(result, full_text.strip())

    def test_extract_description_falls_back_to_expandable_text(self):
        body = "Bilen er nybesiktiget og har full servicehistorikk hos Tesla. " * 5
        html = f"""
        <html><body>
          <div data-testid="expandable-text">{body}</div>
        </body></html>
        """
        result = extract_description_from_html(html)
        self.assertIn("servicehistorikk", result)

    def test_extract_description_falls_back_to_meta_when_nothing_else(self):
        html = '<html><head><meta name="description" content="kort tekst"></head></html>'
        self.assertEqual(extract_description_from_html(html), "kort tekst")


if __name__ == "__main__":
    unittest.main()

