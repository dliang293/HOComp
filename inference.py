import argparse
import os
import torch
from PIL import Image
from hocomp_pipeline import HOCompPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="HOComp Inference Script")
    
    # --- Model Paths ---
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--lora_path", type=str, default="./checkpoints/hocomp_v1.safetensors")
    
    # 1. Background Image Path
    parser.add_argument("--bg_path", type=str, default="path/to/your/background.jpg",
                        help="Path to the background image containing a human subject")
    
    # 2. Foreground Image Path
    parser.add_argument("--fg_path", type=str, default="path/to/your/object.png",
                        help="Path to the foreground object image")
    
    parser.add_argument("--prompt", type=str, 
                        default="A young man holding a vintage camera",
                        )
    parser.add_argument("--box", type=int, nargs=4, default=[300, 300, 700, 700])

    return parser.parse_args()

def load_image_data(path, label, size=(1024, 1024)):

    
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    else:
        # Fallback for code demonstration so it doesn't crash
        return Image.new("RGB", size, (200, 200, 200))

def main():
    args = parse_args()
    
    # 1. Initialize Pipeline
    pipeline = HOCompPipeline(
        base_model_id=args.base_model,
        local_lora_path=args.lora_path
    )
    
    # 2. Load Visual Inputs
    bg_img = load_image_data(args.bg_path, "Background")
    fg_img = load_image_data(args.fg_path, "Foreground", size=(512, 512))
    
    # 3. Load Semantic & Spatial Inputs (Prompt & Box)
    
    # 4. Run Sequence Concatenation Pipeline
    result = pipeline(
        prompt=args.prompt,
        bg_img=bg_img,
        fg_img=fg_img,
        box=args.box
    )
    
    # 5. Save Output
    output_path = "output_hocomp.png"
    result.save(output_path)

if __name__ == "__main__":
    main()