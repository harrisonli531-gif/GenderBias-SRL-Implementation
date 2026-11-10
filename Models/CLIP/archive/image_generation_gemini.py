import os
import csv
import io
import time
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError

def setup_pipeline_directories():
    """Ensures input and output master folders exist on disk and initializes Gemini."""
    input_dir = Path("./input_images")
    output_master_dir = Path("./output_images")
    
    input_dir.mkdir(exist_ok=True)
    output_master_dir.mkdir(exist_ok=True)
    
    # Verify API Key is in the environment
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("❌ Error: GEMINI_API_KEY environment variable not found. Run 'export GEMINI_API_KEY=...' first.")
        
    # Standard SDK Initialization (Handles both legacy and modern AQ. keys)
    client = genai.Client()
    print("✅ Environment initialized. Connected securely to Google GenAI Cloud Ecosystem.")
    return input_dir, output_master_dir, client

def run_nested_batch_pipeline():
    """Processes every image in a folder against every prompt in a CSV using Google Gemini & Imagen."""
    try:
        input_dir, output_master_dir, client = setup_pipeline_directories()
        csv_path = Path("prompts.csv")
        
        if not csv_path.exists():
            print(f"❌ Error: The 'prompts.csv' file was not found at {csv_path.resolve()}")
            return
            
        # 1. Gather all target image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"⚠️ No images found in '{input_dir.resolve()}'. Please drop some photos in there and re-run!")
            return
            
        # 2. Read and cache the list of prompts from the CSV
        prompts_list = []
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row_clean = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                if 'prompt_id' in row_clean and 'prompt' in row_clean:
                    prompts_list.append({
                        'id': row_clean['prompt_id'],
                        'text': row_clean['prompt']
                    })
                
        if not prompts_list:
            print("⚠️ The prompts.csv file is empty or missing required headers.")
            return

        print(f"📊 Target Matrix: {len(image_files)} image(s) × {len(prompts_list)} prompt(s) = {len(image_files) * len(prompts_list)} total variations.")

        # 3. Outer Loop: Process each local image sequentially
        for img_path in image_files:
            img_stem = img_path.stem  
            print(f"\n📂 [STARTING IMAGE GROUP] Processing source picture: {img_path.name}")
            
            # Create a dedicated output subdirectory for this specific input picture
            image_output_folder = output_master_dir / img_stem
            image_output_folder.mkdir(exist_ok=True)
            
            # Step A: Open image using Pillow
            raw_image = Image.open(img_path)
            
            # 4. Inner Loop: Process every prompt against the current image context
            for index, prompt_item in enumerate(prompts_list, start=1):
                p_id = prompt_item['id']
                user_prompt = prompt_item['text']
                
                print(f"  🎬 [{index}/{len(prompts_list)}] Generating Cloud Variant: '{p_id}'...")
                
                try:
                    # Step B: Use Gemini 3.8 Flash to interpret the image layout and modification instruction
                    analysis_prompt = (
                        f"Look at the attached image. The user wants to apply this change: '{user_prompt}'. "
                        f"Combine the visual structure of the attached image with the user's intent to "
                        f"write an ultra-detailed, descriptive prompt for an image generator. Output ONLY the prompt text."
                    )
                    
                    analysis_response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=[analysis_prompt, raw_image]
                    )
                    refined_prompt = analysis_response.text.strip()
                    
                    # Step C: Feed the engineered prompt directly into Imagen 3
                    generation_response = client.models.generate_images(
                        model='imagen-3.0-generate-002',
                        prompt=refined_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            output_mime_type="image/jpeg",
                            aspect_ratio="1:1"
                        )
                    )
                    
                    # Step D: Extract binary image data and save it to disk
                    if generation_response.generated_images:
                        generated_image_obj = generation_response.generated_images
                        image_bytes = generated_image_obj.image.image_bytes
                        
                        final_image = Image.open(io.BytesIO(image_bytes))
                        final_filename = f"{img_stem}_{p_id}.jpg"
                        final_path = image_output_folder / final_filename
                        final_image.save(final_path)
                        
                        print(f"    💾 Saved successfully -> {final_path.resolve()}")
                    else:
                        print(f"    ❌ Error: Imagen completed but returned an empty asset frame for '{p_id}'")
                        
                    # ⚡ Throttling Pause: 1-second delay to preserve stability and avoid free/low-tier API rate spikes
                    time.sleep(1.0)
                    
                except APIError as api_err:
                    print(f"    ❌ Google API Error running prompt '{p_id}': {api_err}")
                    if api_err.code == 429:
                        print("    ⚠️ Rate limit hit. Backing off for 10 seconds...")
                        time.sleep(10.0)
                except Exception as inner_err:
                    print(f"    ❌ Unexpected Error running prompt '{p_id}': {inner_err}")
                    print("    Skipping directly to the next variation...")
                
        print("\n🏁 All nested matrix operations completed successfully using Gemini Cloud Infrastructure!")
            
    except Exception as e:
        print(f"💥 Batch pipeline matrix execution failed: {e}")

if __name__ == "__main__":
    run_nested_batch_pipeline()
