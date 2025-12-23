"""
Real MOO Code Acceptance Tests

Tests that real MOO code from .db files parses without errors.
Uses the lambdamoo_db library to extract verb code.
"""
import os
import pytest
from pathlib import Path

# Import parser
from moo_interp.moo_ast import parse

# Try to import lambdamoo_db
try:
    from lambdamoo_db.reader import load
    HAS_LAMBDAMOO_DB = True
except ImportError:
    HAS_LAMBDAMOO_DB = False


# Path to test databases
DB_DIR = Path(__file__).parent.parent
TOASTCORE_DB = DB_DIR / "toastcore.db"
MINIMAL_DB = DB_DIR / "minimal.db"


def get_verbs_from_db(db_path, limit=None):
    """Extract verb code from a LambdaMOO database file."""
    if not HAS_LAMBDAMOO_DB:
        pytest.skip("lambdamoo_db not installed")
        return []

    if not db_path.exists():
        pytest.skip(f"Database not found: {db_path}")
        return []

    database = load(str(db_path))
    verbs = []

    for verb in database.all_verbs():
        if verb.code:
            code = "\n".join(verb.code)
            obj = database.objects.get(verb.object)
            obj_name = obj.name if obj else f"#{verb.object}"
            verb_ref = f"{obj_name}:{verb.name}"
            verbs.append((code, verb_ref))

            if limit and len(verbs) >= limit:
                break

    return verbs


# =============================================================================
# TOASTCORE DATABASE TESTS
# =============================================================================

# Cache the verbs to avoid reloading for each test
_toastcore_verbs = None


def get_toastcore_verbs(limit=100):
    """Get verbs from toastcore.db, cached."""
    global _toastcore_verbs
    if _toastcore_verbs is None:
        _toastcore_verbs = get_verbs_from_db(TOASTCORE_DB, limit=limit)
    return _toastcore_verbs


@pytest.fixture(scope="module")
def toastcore_verbs():
    """Fixture to provide toastcore verbs."""
    return get_toastcore_verbs(limit=100)


class TestToastcoreDb:
    """Test parsing real verbs from toastcore.db"""

    def test_db_loads(self):
        """Verify we can load the database."""
        if not HAS_LAMBDAMOO_DB:
            pytest.skip("lambdamoo_db not installed")
        if not TOASTCORE_DB.exists():
            pytest.skip(f"Database not found: {TOASTCORE_DB}")

        database = load(str(TOASTCORE_DB))
        assert database is not None

    def test_verbs_exist(self, toastcore_verbs):
        """Verify we extracted some verbs."""
        assert len(toastcore_verbs) > 0, "No verbs found in database"

    @pytest.mark.parametrize("code,verb_ref", get_toastcore_verbs(limit=50))
    def test_verb_parses(self, code, verb_ref):
        """Test that verb code parses without error."""
        try:
            ast = parse(code)
            assert ast is not None
        except Exception as e:
            # Record the failure with useful context
            pytest.fail(f"Failed to parse {verb_ref}:\n{code[:500]}\n\nError: {e}")


# =============================================================================
# MINIMAL DATABASE TESTS
# =============================================================================

@pytest.fixture(scope="module")
def minimal_verbs():
    """Fixture to provide minimal.db verbs."""
    return get_verbs_from_db(MINIMAL_DB, limit=50)


class TestMinimalDb:
    """Test parsing verbs from minimal.db"""

    @pytest.mark.parametrize("code,verb_ref", get_verbs_from_db(MINIMAL_DB, limit=20))
    def test_verb_parses(self, code, verb_ref):
        """Test that verb code parses without error."""
        try:
            ast = parse(code)
            assert ast is not None
        except Exception as e:
            pytest.fail(f"Failed to parse {verb_ref}:\n{code[:500]}\n\nError: {e}")


# =============================================================================
# STATISTICS AND COVERAGE
# =============================================================================

def test_parse_coverage_report():
    """
    Parse all verbs and report coverage statistics.
    Not a real test - just generates a report.
    """
    if not HAS_LAMBDAMOO_DB:
        pytest.skip("lambdamoo_db not installed")
    if not TOASTCORE_DB.exists():
        pytest.skip(f"Database not found: {TOASTCORE_DB}")

    database = load(str(TOASTCORE_DB))
    total = 0
    passed = 0
    failed = 0
    errors = {}

    for verb in database.all_verbs():
        if not verb.code:
            continue

        total += 1
        code = "\n".join(verb.code)

        try:
            ast = parse(code)
            if ast is not None:
                passed += 1
        except Exception as e:
            failed += 1
            error_type = type(e).__name__
            error_msg = str(e)[:100]
            key = f"{error_type}: {error_msg}"
            if key not in errors:
                errors[key] = 0
            errors[key] += 1

    # Print report
    print(f"\n{'='*60}")
    print(f"PARSE COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"Total verbs: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")

    if errors:
        print(f"\nTop errors:")
        for error, count in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:4d}x {error[:70]}")

    print(f"{'='*60}\n")

    # This test always passes - it's just for reporting
    assert True


# =============================================================================
# SPECIFIC KNOWN PATTERNS FROM REAL CODE
# =============================================================================

# These are patterns extracted from real MOO verbs that may be tricky

REAL_WORLD_PATTERNS = [
    # From typical verb code
    ('player:tell("Hello, world!");', "player:tell"),

    # Property chains
    ("return this.location.name;", "property chain"),

    # Error handling pattern (catch expression ends with ' not `)
    ("return `this.prop ! E_PROPNF => \"\"';", "catch property not found"),

    # List operations
    ("return {@this.contents, object};", "splice in list"),

    # Conditional expressions
    ("return condition ? trueval | falseval;", "ternary in return"),

    # Verb calls with complex args
    ('player:tell(tostr("Value: ", value));', "nested function call"),

    # Assignment in condition (if MOO supports it)
    # ("if ((x = func()) != 0) return x; endif", "assignment in condition"),

    # Comparison chains
    ("return 1 <= x && x <= 10;", "range check pattern"),
]

@pytest.mark.parametrize("code,desc", REAL_WORLD_PATTERNS)
def test_real_world_pattern(code, desc):
    """Test common real-world MOO patterns."""
    try:
        ast = parse(code)
        assert ast is not None
    except Exception as e:
        pytest.fail(f"Failed to parse {desc}: {code}\n\nError: {e}")
