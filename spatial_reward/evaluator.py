"""
Core evaluator for SpatialReward.

Consolidates object detection, orientation prediction, depth estimation,
OCR, and CLIP color classification into a unified evaluation pipeline.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Dict, Any, List, Tuple, Optional, Union
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, pipeline
from huggingface_hub import hf_hub_download

from .config import SpatialRewardConfig
from .constants import COLORS, ASSETS_DIR
from .prompt_parser import PromptParser
from .task_classifier import TaskClassifier, ExpertRouter
from .models.vision_tower import DINOv2_MLP
from .models.orientation import get_3angle
from .experts.counting import evaluate_reward
from .experts.position import evaluate_complex_positions
from .experts.orientation import evaluate_orientation as _evaluate_orientation
from .experts.depth import evaluate_3d_relation as _evaluate_3d_relation
from .experts.text import evaluate_text, evaluate_text_with_position


class SpatialRewardEvaluator:
    """
    Multi-dimensional spatial reward evaluator for text-to-image generation.

    Evaluates alignment between generated images and text prompts across:
    - Object counting and presence
    - Color accuracy (via CLIP)
    - 2D spatial positioning (left/right/above/below)
    - 3D depth relations (in front of/behind/inside)
    - Object orientation (facing direction)
    - Text/OCR content on objects

    Args:
        config: SpatialRewardConfig instance, or None to use defaults with env overrides
        device: Device override (shortcut for config.device)
        qwen_model_path: QwenVL model path override (shortcut for config.qwen_model)
    """

    def __init__(
        self,
        config: Optional[SpatialRewardConfig] = None,
        device: Optional[str] = None,
        qwen_model_path: Optional[str] = None,
    ):
        if config is None:
            config = SpatialRewardConfig.from_env()
        if device is not None:
            config.device = device
        if qwen_model_path is not None:
            config.qwen_model = qwen_model_path

        self.config = config
        self.device = config.device

        # Lazy-loaded models
        self._object_detector = None
        self._image_processor = None
        self._orientation_model = None
        self._orientation_preprocess = None
        self._depth_pipe = None
        self._ocr = None
        self._clip_model = None
        self._clip_tokenizer = None
        self._clip_transform = None

        # Prompt parsers
        self.rule_parser = PromptParser()
        self._qwen_parser = None

        # Expert router
        self.expert_router = ExpertRouter(self)

    # ======================== Model Loading ========================

    def _ensure_detector(self):
        if self._object_detector is not None:
            return
        model_id = self.config.grounding_dino_model
        print(f"[INFO] Loading GroundingDINO from: {model_id}")
        self._image_processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=self.config.cache_dir
        )
        self._object_detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=self.config.cache_dir
        ).to(self.device).eval()

    def _ensure_orientation_model(self):
        if self._orientation_model is not None:
            return
        ckpt_path = hf_hub_download(
            repo_id=self.config.orient_repo,
            filename=self.config.orient_ckpt,
            repo_type="model",
            cache_dir=self.config.cache_dir,
            resume_download=True,
        )
        self._orientation_model = DINOv2_MLP(
            dino_mode="large",
            in_dim=1024,
            out_dim=360 + 180 + 180 + 2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
            cache_dir=self.config.cache_dir,
        ).eval().to(self.device)
        self._orientation_model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")
        )
        self._orientation_preprocess = AutoImageProcessor.from_pretrained(
            self.config.dinov2_model, cache_dir=self.config.cache_dir
        )
        print("[INFO] Orientation model loaded")

    def _ensure_depth_model(self):
        if self._depth_pipe is not None:
            return
        self._depth_pipe = pipeline(
            task="depth-estimation",
            model=self.config.depth_model,
            device=self.device,
        )
        print("[INFO] Depth model loaded")

    def _ensure_ocr(self):
        if self._ocr is not None:
            return
        if not self.config.use_ocr:
            return
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=(self.device != "cpu"),
                show_log=False,
            )
            print("[INFO] PaddleOCR loaded")
        except ImportError:
            print("[WARN] paddleocr not installed. Install with: pip install 'spatial-reward[ocr]'")
        except Exception as e:
            print(f"[WARN] Failed to load PaddleOCR: {e}")

    def _ensure_clip(self):
        if self._clip_model is not None:
            return
        if not self.config.use_clip_color:
            return
        try:
            import open_clip
            clip_arch = "ViT-L-14"
            self._clip_model, _, self._clip_transform = open_clip.create_model_and_transforms(
                clip_arch, pretrained="openai", device=self.device
            )
            self._clip_tokenizer = open_clip.get_tokenizer(clip_arch)
            self._clip_model.eval()
            print("[INFO] CLIP model loaded for color classification")
        except Exception as e:
            print(f"[WARN] Failed to load CLIP: {e}")

    def _ensure_qwen_parser(self):
        if self._qwen_parser is not None:
            return
        if self.config.qwen_model is None:
            return
        try:
            from .qwen_parser import create_qwen_parser
            self._qwen_parser = create_qwen_parser(self.config.qwen_model, self.device)
            print(f"[INFO] QwenVL parser loaded")
        except Exception as e:
            print(f"[WARN] Failed to load QwenVL parser: {e}")

    # ======================== Color Classification ========================

    def color_classification(self, image, bboxes, classname):
        """Classify colors of detected objects using CLIP."""
        self._ensure_clip()
        if self._clip_model is None:
            return [None] * len(bboxes)

        from clip_benchmark.metrics import zeroshot_classification as zsc
        from .utils.image import ImageCrops

        clf = zsc.zero_shot_classifier(
            self._clip_model, self._clip_tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ],
            str(self.device),
        )
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes, self._clip_transform),
            batch_size=16, num_workers=0,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self._clip_model, clf, dataloader, str(self.device))
            return [COLORS[index.item()] for index in pred.argmax(1)]

    # ======================== Expert Methods ========================

    def evaluate_reward(self, image, objects, metadata):
        """Evaluate counting, color, and basic position."""
        color_fn = self.color_classification if self.config.use_clip_color else None
        return evaluate_reward(
            image, objects, metadata,
            soft_confidence=self.config.soft_confidence,
            color_classifier_fn=color_fn,
        )

    def evaluate_complex_positions(self, image, objects, metadata):
        """Evaluate multi-object 2D position relations."""
        return evaluate_complex_positions(image, objects, metadata)

    def evaluate_orientation(self, image, objects, metadata):
        """Evaluate object orientation."""
        self._ensure_orientation_model()
        return _evaluate_orientation(
            image, objects, metadata,
            orientation_model=self._orientation_model,
            orientation_preprocess=self._orientation_preprocess,
            device=self.device,
        )

    def evaluate_3d_relation(self, image, objects, metadata):
        """Evaluate 3D depth relations."""
        self._ensure_depth_model()
        return _evaluate_3d_relation(image, objects, metadata, depth_pipe=self._depth_pipe)

    def evaluate_text(self, image, objects, metadata):
        """Evaluate text content on objects."""
        self._ensure_ocr()
        return evaluate_text(image, objects, metadata, ocr_engine=self._ocr)

    def evaluate_text_with_position(self, image, objects, metadata):
        """Joint evaluation: position + text."""
        self._ensure_ocr()
        return evaluate_text_with_position(
            image, objects, metadata,
            ocr_engine=self._ocr,
            position_eval_fn=self.evaluate_reward,
        )

    # ======================== Detection ========================

    @torch.no_grad()
    def _inference_detector(self, image_pils, metadatas):
        """Run GroundingDINO object detection."""
        self._ensure_detector()
        results = []
        for im, metadata in zip(image_pils, metadatas):
            classnames = [m["class"] for m in metadata["include"]]
            text_prompt = " ".join([f"a {n}." for n in classnames])
            inputs = self._image_processor(
                images=im, text=text_prompt, return_tensors="pt"
            ).to(self.device)
            outputs = self._object_detector(**inputs)
            result = self._image_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=[im.size[::-1]],
            )[0]
            results.append(result)
        return results

    # ======================== Prompt Parsing ========================

    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a prompt into metadata. Uses QwenVL if available, else rule-based."""
        if self.config.qwen_model is not None:
            self._ensure_qwen_parser()
            if self._qwen_parser is not None:
                try:
                    metadata = self._qwen_parser.parse(prompt, use_model=True)
                    return metadata
                except Exception as e:
                    print(f"[WARN] QwenVL parsing failed: {e}, using rule-based parser")
        return self.rule_parser.parse(prompt)

    # ======================== Public API ========================

    def evaluate(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompts: Union[str, List[str]],
    ) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
        """
        Evaluate image-prompt alignment.

        Args:
            images: Single image path/PIL or list of images
            prompts: Single prompt string or list of prompts

        Returns:
            Tuple of (scores, rewards, results):
                - scores: List of float scores [0, 1]
                - rewards: List of binary rewards (1.0 if correct, 0.0 if not)
                - results: List of detailed result dicts
        """
        # Normalize inputs
        if isinstance(images, (str, Image.Image)):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]

        # Load images
        image_pils = []
        for img in images:
            if isinstance(img, str):
                image_pils.append(Image.open(img).convert("RGB"))
            else:
                image_pils.append(img.convert("RGB"))

        # Parse prompts to metadata
        metadatas = [self._parse_prompt(p) for p in prompts]

        # Run detection
        det_results = self._inference_detector(image_pils, metadatas)

        # Process each image
        ret = []
        for result, image_pil, metadata, prompt in zip(det_results, image_pils, metadatas, prompts):
            detected = {}
            metadata_classes = [e["class"] for e in metadata["include"]]

            for score_box, classname, box in zip(
                result["scores"].cpu(), result["text_labels"], result["boxes"].cpu()
            ):
                classname = classname.replace("a ", "").replace(".", "").strip()
                if classname not in metadata_classes:
                    for mn in metadata_classes:
                        if classname in mn.split(" "):
                            classname = mn
                            break
                detected.setdefault(classname, []).append((box.numpy(), score_box))

            for classname in detected:
                detected[classname] = detected[classname][: self.config.max_objects]

            image = ImageOps.exif_transpose(image_pil)

            # Route to expert
            is_correct, score = self.expert_router.route(image, detected, metadata)

            ret.append({
                "tag": metadata.get("tag", "basic"),
                "prompt": prompt,
                "correct": is_correct,
                "score": score,
                "metadata": json.dumps(metadata),
            })

        scores = [r["score"] for r in ret]
        rewards = [1.0 if r["correct"] else 0.0 for r in ret]
        return scores, rewards, ret

    def evaluate_with_metadata(
        self,
        images: Union[Image.Image, List[Image.Image]],
        metadatas: Union[Dict, List[Dict]],
    ) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
        """
        Evaluate with pre-built metadata (skip prompt parsing).

        Args:
            images: Single or list of PIL images
            metadatas: Single or list of metadata dicts

        Returns:
            Same as evaluate()
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(metadatas, dict):
            metadatas = [metadatas]

        image_pils = [img.convert("RGB") for img in images]
        det_results = self._inference_detector(image_pils, metadatas)

        ret = []
        for result, image_pil, metadata in zip(det_results, image_pils, metadatas):
            detected = {}
            metadata_classes = [e["class"] for e in metadata["include"]]

            for score_box, classname, box in zip(
                result["scores"].cpu(), result["text_labels"], result["boxes"].cpu()
            ):
                classname = classname.replace("a ", "").replace(".", "").strip()
                if classname not in metadata_classes:
                    for mn in metadata_classes:
                        if classname in mn.split(" "):
                            classname = mn
                            break
                detected.setdefault(classname, []).append((box.numpy(), score_box))

            for classname in detected:
                detected[classname] = detected[classname][: self.config.max_objects]

            image = ImageOps.exif_transpose(image_pil)

            tag = metadata.get("tag", "basic")
            if tag == "complex-multi-pos":
                is_correct, score = self.evaluate_complex_positions(image, detected, metadata)
            elif tag == "orientation":
                is_correct, score = self.evaluate_orientation(image, detected, metadata)
            elif tag == "3d-relation":
                is_correct, score = self.evaluate_3d_relation(image, detected, metadata)
            elif tag in ("text-counting", "text"):
                is_correct, score = self.evaluate_text(image, detected, metadata)
            elif tag == "text-position":
                is_correct, score = self.evaluate_text_with_position(image, detected, metadata)
            else:
                is_correct, score = self.evaluate_reward(image, detected, metadata)

            ret.append({
                "tag": tag,
                "prompt": metadata.get("prompt", ""),
                "correct": is_correct,
                "score": score,
                "metadata": json.dumps(metadata),
            })

        scores = [r["score"] for r in ret]
        rewards = [1.0 if r["correct"] else 0.0 for r in ret]
        return scores, rewards, ret

    def __call__(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompts: Union[str, List[str]],
    ) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
        """Callable interface, same as evaluate()."""
        return self.evaluate(images, prompts)
