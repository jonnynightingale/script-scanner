from pathlib import Path
import json

import cv2
import pytest

from script_ocr import load_character_mapping, script_image_to_json

# Load the character mapping once before any tests run.
character_mapping = load_character_mapping(Path(__file__).parent.parent / "characters.tsv")

TEST_DATA_DIR = Path(__file__).parent / "test_data"

# (image filename, expected json filename)
TEST_CASES = [
    #("trouble_brewing.png", "trouble_brewing.json")
    ("rotting_moors.png", "rotting_moors.json")
]


@pytest.mark.parametrize("image_file,expected_file", TEST_CASES)
def test_script_image_to_json(image_file, expected_file):
    image_path = TEST_DATA_DIR / image_file
    expected_path = TEST_DATA_DIR / expected_file

    image = cv2.imread(str(image_path))
    assert image is not None, f"Failed to load image: {image_path}"

    _, _, actual_json = script_image_to_json(character_mapping, image)

    with expected_path.open("r", encoding="utf-8") as f:
        expected_json = json.load(f)

    actual_json = json.loads(actual_json)

    assert actual_json == expected_json
