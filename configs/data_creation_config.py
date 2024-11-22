from dataclasses import dataclass, field
from typing import List


@dataclass
class DataCreationConfig:
    prompts: List[str] = field(default_factory=lambda: [
        "A 3d rendering of a humanoid cactus wearing a cowboy hat playing poker, white background",

    ])
    objects_to_detect: List[str] = field(default_factory=lambda: [
        ["cactus", "cowboy hat", "poker"],
    ])
    num_images_per_prompt: int = 2
    output_dir: str = "/home/dcor/orlichter/concept_collage/data"