"""Tests for the rule-based prompt parser."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from spatial_reward.prompt_parser import PromptParser, parse_prompt_to_metadata


def test_basic_counting():
    metadata = parse_prompt_to_metadata("three dogs")
    assert metadata["tag"] == "basic"
    assert len(metadata["include"]) >= 1
    dogs = [e for e in metadata["include"] if e["class"] == "dogs"]
    assert len(dogs) == 1
    assert dogs[0]["count"] == 3


def test_color_detection():
    metadata = parse_prompt_to_metadata("a red car")
    assert any(e.get("color") == "red" for e in metadata["include"])


def test_position_2d():
    metadata = parse_prompt_to_metadata("a cat to the left of a dog")
    assert metadata["tag"] in ("position", "complex-multi-pos")
    # Should have position relation in include
    has_position = any("position" in e for e in metadata["include"])
    assert has_position


def test_orientation():
    metadata = parse_prompt_to_metadata("a horse facing right")
    assert metadata["tag"] == "orientation"
    has_orientation = any("orientation" in e for e in metadata["include"])
    assert has_orientation


def test_3d_relation():
    metadata = parse_prompt_to_metadata("a car in front of a building")
    assert metadata["tag"] == "3d-relation"
    has_3d = any("position3d" in e for e in metadata["include"])
    assert has_3d


def test_multiple_objects():
    metadata = parse_prompt_to_metadata("two cats and a dog")
    classes = [e["class"] for e in metadata["include"]]
    assert "cats" in classes or "dog" in classes


def test_complex_positions():
    metadata = parse_prompt_to_metadata(
        "a vase to the left of a clock, the clock above a tie, the tie to the right of a book"
    )
    assert metadata["tag"] == "complex-multi-pos"


def test_prompt_field_preserved():
    prompt = "a blue ball above a red box"
    metadata = parse_prompt_to_metadata(prompt)
    assert metadata["prompt"] == prompt


if __name__ == "__main__":
    tests = [
        test_basic_counting,
        test_color_detection,
        test_position_2d,
        test_orientation,
        test_3d_relation,
        test_multiple_objects,
        test_complex_positions,
        test_prompt_field_preserved,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} - {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__} - {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
