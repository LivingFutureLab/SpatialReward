from typing import Dict, Any, List, Callable
from enum import Enum


class TaskType(Enum):
    BASIC_COUNT = "basic"
    COLOR_CHECK = "color"
    POSITION_2D = "position"
    POSITION_3D = "3d-relation"
    ORIENTATION = "orientation"
    COMPLEX_MULTI_POS = "complex-multi-pos"
    TEXT = "text"


class TaskClassifier:
    """Classifies metadata into task types for expert routing."""

    def __init__(self):
        self.tag_to_task_type = {
            "basic": TaskType.BASIC_COUNT,
            "color": TaskType.COLOR_CHECK,
            "position": TaskType.POSITION_2D,
            "3d-relation": TaskType.POSITION_3D,
            "orientation": TaskType.ORIENTATION,
            "complex-multi-pos": TaskType.COMPLEX_MULTI_POS,
            "text": TaskType.TEXT,
            "text-counting": TaskType.TEXT,
            "text-position": TaskType.TEXT,
        }

    def classify(self, metadata: Dict[str, Any]) -> TaskType:
        tag = metadata.get("tag", "basic")
        task_type = self.tag_to_task_type.get(tag, TaskType.BASIC_COUNT)

        # Override: if any include entry has a 'text' field, route to TEXT
        has_text = any("text" in req for req in metadata.get("include", []))
        if has_text:
            return TaskType.TEXT

        return task_type

    def get_expert_name(self, task_type: TaskType) -> str:
        expert_map = {
            TaskType.BASIC_COUNT: "evaluate_reward",
            TaskType.COLOR_CHECK: "evaluate_reward",
            TaskType.POSITION_2D: "evaluate_reward",
            TaskType.POSITION_3D: "evaluate_3d_relation",
            TaskType.ORIENTATION: "evaluate_orientation",
            TaskType.COMPLEX_MULTI_POS: "evaluate_complex_positions",
            TaskType.TEXT: "evaluate_text_with_position",
        }
        return expert_map.get(task_type, "evaluate_reward")


class ExpertRouter:
    """Routes evaluation tasks to the appropriate expert method."""

    TASK_METHOD_MAP: Dict[TaskType, str] = {
        TaskType.BASIC_COUNT: "evaluate_reward",
        TaskType.COLOR_CHECK: "evaluate_reward",
        TaskType.POSITION_2D: "evaluate_reward",
        TaskType.POSITION_3D: "evaluate_3d_relation",
        TaskType.ORIENTATION: "evaluate_orientation",
        TaskType.COMPLEX_MULTI_POS: "evaluate_complex_positions",
        TaskType.TEXT: "evaluate_text_with_position",
    }

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.classifier = TaskClassifier()
        self._validate_evaluator()
        self.expert_methods = self._build_expert_methods()

    def _validate_evaluator(self):
        required_methods = set(self.TASK_METHOD_MAP.values())
        missing = [m for m in required_methods if not hasattr(self.evaluator, m)]
        if missing:
            available = [
                m for m in dir(self.evaluator)
                if callable(getattr(self.evaluator, m)) and not m.startswith("_")
            ]
            raise AttributeError(
                f"[ExpertRouter] Evaluator missing methods: {missing}\n"
                f"Available: {available}"
            )

    def _build_expert_methods(self) -> Dict[TaskType, Callable]:
        expert_methods = {}
        for task_type, method_name in self.TASK_METHOD_MAP.items():
            method = getattr(self.evaluator, method_name, None)
            if method is None:
                method = self.evaluator.evaluate_reward
            expert_methods[task_type] = method
        return expert_methods

    def route(self, image, objects: Dict, metadata: Dict[str, Any]) -> tuple:
        task_type = self.classifier.classify(metadata)
        expert_method = self.expert_methods.get(task_type, self.evaluator.evaluate_reward)
        print(f"[ExpertRouter] Routing to: {expert_method.__name__} (task: {task_type.value})")
        return expert_method(image, objects, metadata)

    def batch_route(
        self, images: List, objects_list: List[Dict], metadatas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        for image, objects, metadata in zip(images, objects_list, metadatas):
            correct, score = self.route(image, objects, metadata)
            results.append({
                "tag": metadata.get("tag", "basic"),
                "prompt": metadata.get("prompt", ""),
                "correct": correct,
                "score": score,
                "metadata": metadata,
            })
        return results
