"""Classes, canonical answers and prompts, read from the data config."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskSpec:
    class_keys: List[str]
    answers: Dict[str, str]           # class key -> canonical answer the model emits
    codes: Dict[str, str]             # BRISC file-name code -> class key
    folders: Dict[str, str]           # classification folder name -> class key
    planes: Dict[str, str]
    seg_token: str
    prompts: Dict[str, str]           # task -> prompt
    task_mix: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, data_cfg: dict) -> "TaskSpec":
        classes = data_cfg["classes"]
        mix = data_cfg.get("task_mix", {"joint": 1.0})
        tot = float(sum(mix.values()))
        return cls(
            class_keys=[c["key"] for c in classes],
            answers={c["key"]: c["answer"] for c in classes},
            codes={c["code"]: c["key"] for c in classes},
            folders={c["folder"]: c["key"] for c in classes},
            planes=dict(data_cfg.get("planes", {})),
            seg_token=data_cfg.get("seg_token", "<SEG>"),
            prompts=dict(data_cfg["prompts"]),
            task_mix={k: v / tot for k, v in mix.items()},
        )

    def idx(self, key: str) -> int:
        return self.class_keys.index(key)

    def candidates(self) -> List[str]:
        return [self.answers[k] for k in self.class_keys]

    def prompt_for(self, task: str) -> str:
        return self.prompts[task]

    def answer_for(self, task: str, class_key: str) -> str:
        if task == "joint":
            return f"{self.answers[class_key]} {self.seg_token}"
        if task == "cls":
            return self.answers[class_key]
        if task == "seg":
            return self.seg_token
        raise ValueError(task)
