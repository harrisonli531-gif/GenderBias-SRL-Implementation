import os
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

def setup_pipeline():
    """Prepares output directories without requiring any web API keys."""
    output_dir = Path("./output_images")
    output_dir.mkdir(exist_ok=True)
    print("✅ Local environment directories prepared.")
    return output_dir

def create_dummy_image_for_testing():
    """Creates a temporary test image if you don't have one."""
    img_path = Path("test_input.jpg")
    if not img_path.exists():
        img = Image.new('RGB', (400, 400), color='blue')
        d = ImageDraw.Draw(img)
        d.text((20, 20), "Input Shape", fill=(255,255,255))
        img.save(img_path)
        print(f"📦 Created a temporary test image at: {img_path}")
    return img_path

def run_fully_local_pipeline():
    """Processes images and prompts natively on your own machine hardware."""
    try:
        output_dir = setup_pipeline()
        input_image_path = create_dummy_image_for_testing()
        
        user_prompt = "Transform this simple blue square into an intricate neon cyber-punk crystal artifact, highly detailed, digital art"
        
        # Determine the fastest available local processing unit
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Computing on local hardware device: {device.upper()}")
        
        # 1. Phase 1: Local Image Captioning (Bypassing Gemini)
        print("🧠 Phase 1: Analyzing input image structure using local BLIP model...")
        raw_image = Image.open(input_image_path).convert('RGB')
        
        # Load the free open-source vision processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        # Extract visual context from your input picture
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = vision_model.generate(**inputs)
        image_caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Merge what the computer sees with your editing instruction
        refined_prompt = f"{user_prompt}, keeping the core geometry layout matching a {image_caption}"
        print(f"✨ Merged Production Prompt:\n'{refined_prompt}'\n")
        
        # 2. Phase 2: Local Image Generation
        print("🤖 Phase 2: Loading local Stable Diffusion engine...")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to(device)
        
        # 3. Phase 3: Generate and Save the Output
        print("🎨 Phase 3: Rendering output image locally (this will take a moment)...")
        generator_output = pipe(refined_prompt).images[0]
        
        final_path = output_dir / "local_cyberpunk_output.png"
        generator_output.save(final_path)
        print(f"🎉 Success! Locally generated asset saved to: {final_path.resolve()}")
            
    except Exception as e:
        print(f"💥 Local pipeline execution failed: {e}")

if __name__ == "__main__":
    print("🏁 Starting Fully Local Machine Pipeline Test...")
    run_fully_local_pipeline()
