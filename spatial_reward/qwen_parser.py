"""
Optional QwenVL-based prompt parser.

This module requires the 'qwen' extra:
    pip install "spatial-reward[qwen]"

It uses Qwen2.5-VL to intelligently parse natural language prompts into
structured metadata, providing better accuracy than the rule-based parser
for complex or ambiguous prompts.
"""

import json
import re
from typing import Dict, Any, Optional

from .prompt_parser import PromptParser


class QwenVLPromptParser:
    """LLM-based prompt parser using Qwen2.5-VL for structured metadata extraction."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.rule_parser = PromptParser()

        self.system_prompt = (
            "You are an expert at analyzing image generation prompts. "
            "Given a text prompt, extract structured metadata in JSON format.\n\n"
            "Output format:\n"
            '{"tag": "<task_type>", "prompt": "<original_prompt>", "include": [...]}\n\n'
            "Task types (tag):\n"
            '- "basic": simple object counting\n'
            '- "color": objects with color requirements\n'
            '- "position": 2D spatial relations (left/right/above/below)\n'
            '- "complex-multi-pos": multiple position relations\n'
            '- "3d-relation": depth relations (in front of/behind/inside)\n'
            '- "orientation": facing direction requirements\n'
            '- "text": text content on objects\n\n'
            "Each include entry: "
            '{"class": "object_name", "count": N, ...optional: "color", "position", "orientation", "position3d", "text"}\n'
            "position format: [relation_string, target_index]\n"
            "position3d format: [relation_string, target_index]\n"
            "orientation format: [direction_string]\n"
            "text format: [expected_text_per_object]\n\n"
            "Only output valid JSON, no explanation."
        )

    def load_model(self):
        """Load the Qwen2.5-VL model. Called lazily on first parse."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
            )
            self.model.eval()
            print(f"[INFO] QwenVL parser model loaded from: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load QwenVL model: {e}")

    def parse(self, prompt: str, use_model: bool = True) -> Dict[str, Any]:
        """
        Parse a prompt into structured metadata.

        Args:
            prompt: Natural language prompt
            use_model: If True, use LLM; if False, use rule-based fallback

        Returns:
            Structured metadata dict
        """
        if use_model:
            try:
                return self._parse_with_model(prompt)
            except Exception as e:
                print(f"[WARN] QwenVL parsing failed: {e}, falling back to rule-based")
                return self._parse_with_rules(prompt)
        return self._parse_with_rules(prompt)

    def _parse_with_model(self, prompt: str) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        from qwen_vl_utils import process_vision_info
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Parse this prompt: {prompt}"},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        metadata = self._extract_json_from_response(response)
        metadata["prompt"] = prompt
        return metadata

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response."""
        # Try direct parse
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {response[:200]}")

    def _parse_with_rules(self, prompt: str) -> Dict[str, Any]:
        return self.rule_parser.parse(prompt)


def create_qwen_parser(model_path: str, device: str = "cuda") -> QwenVLPromptParser:
    """Create and load a QwenVL parser instance."""
    parser = QwenVLPromptParser(model_path, device)
    parser.load_model()
    return parser
