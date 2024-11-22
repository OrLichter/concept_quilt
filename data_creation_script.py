import pyrallis
import cv2

from tqdm import tqdm
from pathlib import Path

from configs.data_creation_config import DataCreationConfig
from data_creation.generate_images import ImageGenerator
from data_creation.collage_creator import CollageCreator
from itertools import permutations


@pyrallis.wrap()
def main(cfg: DataCreationConfig):
    base_output_dir = Path(cfg.output_dir)
    images_generator = ImageGenerator()
    collage_creator = CollageCreator()
    pbar = tqdm(zip(cfg.prompts, cfg.objects_to_detect), desc="Creating collages")
    for prompt, objects in pbar:
        pbar.set_postfix_str(prompt)
        output_dir = base_output_dir / prompt
        output_dir.mkdir(parents=True, exist_ok=True)
        images = images_generator.generate_images(prompt, cfg.num_images_per_prompt - 1)
        image_pairs = list(permutations(images, 2))
        for i, pair in enumerate(image_pairs):
            collage_image = collage_creator.create_collage(source_image=pair[0], target_image=pair[1], objects_to_detect=objects)
            print(f"Saving images for prompt {prompt}")
            pair[0].save(output_dir / f"{i}_source.png")
            pair[1].save(output_dir / f"{i}_target.png")
            cv2.imwrite(str(output_dir / f"{i}_collage.png"), collage_image[..., ::-1])


if __name__ == "__main__":
    main()