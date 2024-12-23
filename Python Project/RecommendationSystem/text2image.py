from diffusers import DiffusionPipeline
import torch
import os

# Corrected path (using raw string for Windows path)
model_path = r"D:\Python Project\RecommendationSystem\stable-diffusion-xl-base-1.0"

# Ensure the directory exists (if not, create it)
os.makedirs(model_path, exist_ok=True)

# Check if the model already exists locally in the specified path
if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    print("Model not found locally. Downloading...")
    # Load and download the model from Hugging Face
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    # Save the model locally in the specified directory for future use
    pipe.save_pretrained(model_path)
else:
    print("Loading model from local path...")
    # Load the model from the saved local path
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

# Move the model to GPU (uncomment this line if you have a compatible GPU)
pipe.to("cpu")

# Optionally enable memory efficient attention (if using torch < 2.0)
# pipe.enable_xformers_memory_efficient_attention()

# Set the prompt for the image generation
prompt = "An astronaut riding a green horse"

# Generate the image based on the prompt
images = pipe(prompt=prompt).images[0]

# Show the generated image
images.show()