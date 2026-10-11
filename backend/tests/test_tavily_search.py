"""Tests for Tavily paper-resource filtering."""

import unittest

from backend.app.services.tavily_search import is_concrete_paper_resource


class ConcretePaperResourceTests(unittest.TestCase):
    def test_accepts_concrete_paper_page(self):
        self.assertTrue(
            is_concrete_paper_resource(
                "Irrigation-nutrient synergy improves productivity",
                "https://www.nature.com/articles/s41598-026-52821-z",
            )
        )
        self.assertTrue(
            is_concrete_paper_resource(
                "A Review of Challenges in Machine Learning",
                "https://arxiv.org/abs/1806.00388",
            )
        )

    def test_accepts_direct_paper_pdf(self):
        self.assertTrue(
            is_concrete_paper_resource(
                "A concrete agricultural research paper",
                "https://example.edu/papers/crop-yield.pdf",
            )
        )

    def test_rejects_article_year_listing(self):
        self.assertFalse(
            is_concrete_paper_resource(
                "Articles in 2026 | npj Sustainable Agriculture",
                "https://www.nature.com/npjsustainagric/articles?year=2026",
            )
        )

    def test_rejects_reference_pdf(self):
        self.assertFalse(
            is_concrete_paper_resource(
                "Irrigation-nutrient synergy improves productivity",
                "https://www.nature.com/articles/s41598-026-52821-z_reference.pdf",
            )
        )

    def test_rejects_invalid_article_in_press_title(self):
        self.assertFalse(
            is_concrete_paper_resource(
                "ARTICLE IN PRESS",
                "https://example.edu/papers/agriculture.pdf",
            )
        )

    def test_rejects_arxiv_listing(self):
        self.assertFalse(
            is_concrete_paper_resource(
                "Machine Learning Mar 2018",
                "https://arxiv.org/list/stat.ML/2018-03",
            )
        )


if __name__ == "__main__":
    unittest.main()
