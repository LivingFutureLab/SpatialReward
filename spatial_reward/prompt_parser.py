import re
import json
from typing import List, Dict, Any, Optional


TASK_TYPES = {
    "basic_count": "basic_count",
    "color_check": "color_check",
    "position_2d": "position_2d",
    "position_3d": "position_3d",
    "orientation": "orientation",
    "complex_multi_pos": "complex_multi_pos",
}

COLOR_WORDS = {
    "red": "red", "orange": "orange", "yellow": "yellow",
    "green": "green", "blue": "blue", "purple": "purple",
    "pink": "pink", "brown": "brown", "black": "black", "white": "white",
}

POSITION_WORDS = {
    "left": "left of", "right": "right of",
    "above": "above", "below": "below",
    "in front of": "in front of", "behind": "behind",
    "inside": "inside",
}

ORIENTATION_WORDS = {
    "facing left": "facing left",
    "facing right": "facing right",
    "facing forward": "facing forward",
    "facing backward": "facing backward",
}


class PromptParser:
    """
    Rule-based parser that converts natural language prompts into structured metadata
    for spatial reward evaluation.
    """

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, re.Pattern]:
        patterns = {}
        # Object counting: "3 red apples", "two cars", "a dog"
        patterns["count_object"] = re.compile(
            r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
            r"((?:red|orange|yellow|green|blue|purple|pink|brown|black|white)\s+)?"
            r"(\w+)",
            re.IGNORECASE,
        )
        # Position relation: "A to the left of B", "A above B"
        patterns["position_rel"] = re.compile(
            r"(\w+(?:\s+\w+)?)\s+"
            r"(?:to the\s+)?(left|right|above|below|in front of|behind)\s+"
            r"(?:of\s+)?(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        # Orientation: "facing left", "facing right"
        patterns["orientation"] = re.compile(
            r"(facing\s+(?:left|right|forward|backward))",
            re.IGNORECASE,
        )
        # 3D relation: "A in front of B", "A behind B"
        patterns["position_3d"] = re.compile(
            r"(\w+(?:\s+\w+)?)\s+"
            r"(in front of|behind|inside)\s+"
            r"(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        return patterns

    def parse(self, prompt: str) -> Dict[str, Any]:
        """
        Parse a natural language prompt into structured metadata.

        Args:
            prompt: e.g. "A vase to the left of a clock, the clock positioned below a tie."

        Returns:
            Standardized metadata dict with 'tag', 'prompt', 'include' fields.
        """
        objects = self._extract_objects(prompt)
        positions = self._extract_positions(prompt)
        orientations = self._extract_orientations(prompt)
        positions_3d = self._extract_3d_positions(prompt)
        task_type = self._classify_task(objects, positions, orientations, positions_3d)
        metadata = self._build_metadata(prompt, objects, positions, orientations, positions_3d, task_type)
        return metadata

    def _extract_objects(self, prompt: str) -> List[Dict[str, Any]]:
        objects = []
        seen = set()
        for match in self.patterns["count_object"].finditer(prompt):
            count_str = match.group(1)
            color = match.group(2).strip().lower() if match.group(2) else None
            classname = match.group(3).lower()
            key = f"{classname}_{color or 'none'}"
            if key in seen:
                continue
            seen.add(key)
            count = self._parse_count(count_str)
            obj = {"class": classname, "count": count}
            if color:
                obj["color"] = color
            objects.append(obj)
        return objects

    def _extract_positions(self, prompt: str) -> List[Dict[str, Any]]:
        positions = []
        for match in self.patterns["position_rel"].finditer(prompt):
            obj_a = self._strip_article(match.group(1).lower())
            relation = match.group(2).lower()
            obj_b = self._strip_article(match.group(3).lower())
            positions.append({"obj_a": obj_a, "relation": relation, "obj_b": obj_b})
        return positions

    def _extract_orientations(self, prompt: str) -> List[Dict[str, Any]]:
        orientations = []
        for match in self.patterns["orientation"].finditer(prompt):
            orientation = match.group(1).lower()
            orientations.append({"orientation": orientation})
        return orientations

    def _extract_3d_positions(self, prompt: str) -> List[Dict[str, Any]]:
        positions_3d = []
        for match in self.patterns["position_3d"].finditer(prompt):
            obj_a = self._strip_article(match.group(1).lower())
            relation = match.group(2).lower()
            obj_b = self._strip_article(match.group(3).lower())
            positions_3d.append({"obj_a": obj_a, "relation": relation, "obj_b": obj_b})
        return positions_3d

    @staticmethod
    def _strip_article(text: str) -> str:
        for article in ("a ", "an ", "the "):
            if text.startswith(article):
                return text[len(article):]
        return text

    def _parse_count(self, count_str: str) -> int:
        count_map = {
            "a": 1, "an": 1, "one": 1,
            "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        count_str = count_str.lower()
        if count_str in count_map:
            return count_map[count_str]
        try:
            return int(count_str)
        except ValueError:
            return 1

    def _classify_task(
        self,
        objects: List[Dict],
        positions: List[Dict],
        orientations: List[Dict],
        positions_3d: List[Dict],
    ) -> str:
        if positions_3d:
            return TASK_TYPES["position_3d"]
        elif orientations:
            return TASK_TYPES["orientation"]
        elif len(positions) > 2:
            return TASK_TYPES["complex_multi_pos"]
        elif positions:
            return TASK_TYPES["position_2d"]
        elif any("color" in obj for obj in objects):
            return TASK_TYPES["color_check"]
        else:
            return TASK_TYPES["basic_count"]

    def _build_metadata(
        self,
        prompt: str,
        objects: List[Dict],
        positions: List[Dict],
        orientations: List[Dict],
        positions_3d: List[Dict],
        task_type: str,
    ) -> Dict[str, Any]:
        include_list = []

        for i, obj in enumerate(objects):
            entry = {"class": obj["class"], "count": obj["count"]}
            if "color" in obj:
                entry["color"] = obj["color"]

            # Position relations
            for pos in positions:
                if pos["obj_a"] == obj["class"]:
                    target_idx = self._find_object_index(objects, pos["obj_b"])
                    if target_idx is not None:
                        entry["position"] = [pos["relation"], target_idx]
                        break

            # Orientation
            for ori in orientations:
                if "orientation" not in entry:
                    entry["orientation"] = [ori["orientation"]]

            # 3D position
            for pos_3d in positions_3d:
                if pos_3d["obj_a"] == obj["class"]:
                    target_idx = self._find_object_index(objects, pos_3d["obj_b"])
                    if target_idx is not None:
                        entry["position3d"] = [pos_3d["relation"], target_idx]
                        break

            include_list.append(entry)

        tag_map = {
            TASK_TYPES["basic_count"]: "basic",
            TASK_TYPES["color_check"]: "color",
            TASK_TYPES["position_2d"]: "position",
            TASK_TYPES["position_3d"]: "3d-relation",
            TASK_TYPES["orientation"]: "orientation",
            TASK_TYPES["complex_multi_pos"]: "complex-multi-pos",
        }

        metadata = {
            "tag": tag_map.get(task_type, "basic"),
            "prompt": prompt,
            "include": include_list,
        }
        return metadata

    def _find_object_index(self, objects: List[Dict], classname: str) -> Optional[int]:
        # Strip common articles
        stripped = classname
        for article in ("a ", "an ", "the "):
            if stripped.startswith(article):
                stripped = stripped[len(article):]
                break

        for i, obj in enumerate(objects):
            obj_class = obj["class"]
            if obj_class == classname or obj_class == stripped:
                return i
            if stripped in obj_class or obj_class in stripped:
                return i
        return None


def parse_prompt_to_metadata(prompt: str) -> Dict[str, Any]:
    """Convenience function: parse a prompt into metadata."""
    parser = PromptParser()
    return parser.parse(prompt)
