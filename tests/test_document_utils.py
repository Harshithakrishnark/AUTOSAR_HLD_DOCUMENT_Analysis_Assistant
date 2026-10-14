from app.document_utils import search_pages


def test_search_pages_finds_matching_page():
    pages = [
        {"page": 1, "text": "The software component has provided ports."},
        {"page": 2, "text": "Signals are routed between modules."},
    ]
    results = search_pages(pages, "provided ports")
    assert results
    assert results[0]["page"] == 1


def test_search_pages_empty_for_unmatched_query():
    pages = [{"page": 1, "text": "Component design overview."}]
    assert search_pages(pages, "quantum bananas") == []
