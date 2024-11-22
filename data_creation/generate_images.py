from diffusers import AutoPipelineForText2Image, DDIMScheduler, LCMScheduler, FluxPipeline
import torch
import matplotlib.pyplot as plt

from typing import Literal
from transformers import CLIPVisionModelWithProjection

class ImageGenerator:
    def __init__(
        self,
        ip_adapter_scale: float = 1,
        device: str = "cuda:0",
        offload_to_cpu: bool = False,
    ):
        self.device= device
        self.dtype = torch.float32 if device == "cpu" else torch.float16

        # Create a pipeline for generating images
        self.generation_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(self.device)

        if offload_to_cpu:
            self.generation_pipeline.enable_model_cpu_offload()

        # Create a pipeline for generating images with IP-Adapter. I think 2 pipelines will be faster than loadeing and unloading IP Adapter
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=self.dtype,
        ).to(self.device)

        self.ip_adapter_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.dtype,
            image_encoder=image_encoder,
        ).to(self.device)

        self.ip_adapter_pipeline.scheduler = DDIMScheduler.from_config(self.ip_adapter_pipeline.scheduler.config)
        self.ip_adapter_pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"]
        )
        self.ip_adapter_pipeline.set_ip_adapter_scale(ip_adapter_scale)
        if offload_to_cpu:
            self.ip_adapter_pipeline.enable_model_cpu_offload()

        self.diffusion_steps = 50
        self.guidance_scale = 5.0
        self.generator = torch.Generator(device="cpu").manual_seed(0)


    def set_generator_base_model(
        self,
        generation_type: Literal["sdxl", "lcm_lora"]
    ):
        if generation_type == "lcm_lora":
            self.ip_adapter_pipeline.scheduler = LCMScheduler.from_config(self.ip_adapter_pipeline.scheduler.config)
            self.ip_adapter_pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            self.diffusion_steps = 4
            self.guidance_scale = 0.0
        if generation_type == "sdxl":
            self.ip_adapter_pipeline.scheduler = DDIMScheduler.from_config(self.ip_adapter_pipeline.scheduler.config)
            self.diffusion_steps = 50
            self.guidance_scale = 5.0
            self.ip_adapter_pipeline.unload_lora_weights()
            
    def generate_images(
        self,
        text: str,
        num_images_per_prompt: int = 1,
    ):
        base_image = self.generation_pipeline(
            prompt=text,
            num_inference_steps=4,  # Always use 4 steps for Flux Scnell
            max_sequence_length=256,
            generator=self.generator,
            num_images_per_prompt=1,
            guidance_scale=0.0,
        ).images[0]
        
        images = self.ip_adapter_pipeline(
            prompt=text,
            ip_adapter_image=[base_image],
            negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            num_inference_steps=self.diffusion_steps,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        
        return [base_image] + images
    

if __name__ == "__main__":
    generator = ImageGenerator(device="cuda:0", offload_to_cpu=True)
    images = generator.generate_images("a 3d render of cactus playing poker wearing a cowboy hat, white background", num_images_per_prompt=2)

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    titles = ["base image"] + [f"IP-A {i}" for i in range(1, len(images))]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("/home/dcor/orlichter/concept_collage/generated_images.jpg")
