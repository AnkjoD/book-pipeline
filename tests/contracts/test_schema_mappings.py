import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def assignment_value(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Không tìm thấy {name} trong {path}")


def test_book_category_mapping_matches_schema_contract():
    expected = {
        1: "Book",
        2: "Novella",
        3: "Short Story",
        4: "Graphic Novel",
        5: "Fan Fiction",
        6: "Research Paper",
        7: "Poetry",
        8: "Collection",
        9: "Web Novel",
        10: "Light Novel",
    }
    path = ROOT / "src/transform/bronze_to_silver/books.py"
    assert assignment_value(path, "_BOOK_CATEGORY") == expected


def test_edition_uses_official_reading_format_lookup():
    path = ROOT / "src/transform/bronze_to_silver/editions.py"
    text = path.read_text(encoding="utf-8")
    assert "_FORMAT" not in text
    assert 'drop("reading_format_id"' not in text

    references = ROOT / "src/ingestion/fetchers/reference_fetchers.py"
    assert "class ReadingFormatsFetcher" in references.read_text(encoding="utf-8")


def test_core_catalog_excludes_optional_platforms():
    cli = ROOT / "src/ingestion/cli.py"
    core = assignment_value(cli, "CORE_ENTITY_KEYS")
    assert "reading_formats" in core
    assert "platforms" not in core


def test_every_core_entity_has_a_silver_transform():
    transform_dir = ROOT / "src/transform/bronze_to_silver"
    for module in [
        "books", "authors", "editions", "series", "tags", "publishers",
        "countries", "languages", "reading_formats",
    ]:
        assert (transform_dir / f"{module}.py").exists()


def test_authors_silver_enforces_one_row_per_id():
    path = ROOT / "src/transform/bronze_to_silver/authors.py"
    assert '.dropDuplicates(["id"])' in path.read_text(encoding="utf-8")


def test_gold_editions_is_scoped_to_valid_gold_books():
    path = ROOT / "src/transform/silver_to_gold/models/dim_editions.sql"
    text = path.read_text(encoding="utf-8")
    assert "select book_id from {{ ref('dim_books') }}" in text
    assert "inner join valid_books" in text


def test_transform_does_not_filter_nonexistent_merged_state():
    transform_dir = ROOT / "src/transform/bronze_to_silver"
    for path in transform_dir.glob("*.py"):
        assert '"merged"' not in path.read_text(encoding="utf-8")


def test_every_queried_entity_exists_in_schema():
    schema = json.loads((ROOT / "hardcover_full_schema.json").read_text(encoding="utf-8"))
    for entity in [
        "Books",
        "Authors",
        "Editions",
        "Series",
        "Tags",
        "Publishers",
        "Countries",
        "Languages",
        "Reading Formats",
        "Platforms",
    ]:
        assert entity in schema
