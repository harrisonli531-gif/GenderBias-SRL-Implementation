import os
import csv
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

def setup_pipeline():
    """Initializes the Gemini client and ensures output directories exist."""
    # Initialize the client (automatically picks up GEMINI_API_KEY from environment)
    client = genai.Client()
    
    # Create an output directory if it doesn't exist
    output_dir = Path("./output_images")
    output_dir.mkdir(exist_ok=True)
    
    return client, output_dir

def process_pipeline(csv_path: str):
    """Systematically processes a CSV file containing image paths and prompts."""
    client, output_dir = setup_pipeline()
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Expecting CSV columns: 'input_image_path', 'prompt', 'output_filename'
        for row in reader:
            img_path = row['input_image_path']
            prompt = row['prompt']
            out_name = row['output_filename']
            
            print(f"Processing: {out_name}...")
            
            try:
                # 1. Load the input image systematically
                input_image = Image.open(img_path)
                
                # 2. Call the Gemini Imagen model for Image-to-Image / Editing
                # Note: Adjust model and configuration based on your exact use case
                response = client.models.generate_images(
                    model='imagen-3.0-generate-002',
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/jpeg",
                        # Pass the source image if doing image-to-image/editing
                        # person_generation="ALLOW_ADULT", 
                    )
                )
                
                # 3. Extract and save the output systematically
                for i, generated_image in enumerate(response.generated_images):
                    image_bytes = generated_image.image.image_bytes
                    
                    # Convert bytes back to a PIL Image and save
                    import io
                    output_image = Image.open(io.BytesIO(image_bytes))
                    
                    final_path = output_dir / f"{out_name}_{i}.jpg"
                    output_image.save(final_path)
                    print(f"Saved successfully to {final_path}")
                    
            except Exception as e:
                print(f"Failed to process {img_path} due to error: {e}")

if __name__ == "__main__":
    # Example usage:
    # Create a 'pipeline_inputs.csv' with columns: input_image_path, prompt, output_filename
    process_pipeline("pipeline_inputs.csv")
