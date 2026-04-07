import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
from pathlib import Path
import torchvision.transforms as transforms

from model import Generator

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path("models")

MODELS = {
    "Model A": "sketchy_G.pth",
    "Model B": "weeb_G.pth",
    "Model C": "sketchyweeb_G.pth"
}

@st.cache_resource
def load_generator(model_name):
    """Load a pre-trained generator model."""
    model_path = MODELS_DIR / MODELS[model_name]
    
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    
    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
    generator.eval()
    
    return generator

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """Convert model output tensor to PIL Image."""
    # denormalize, so it doesnt look weird (just undo normalization steps)
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    tensor = (tensor * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(tensor)

def transform_image(generator: Generator, image_tensor: torch.Tensor) -> Image.Image:
    """Transform image using the generator."""
    with torch.no_grad():
        output_tensor = generator(image_tensor)
    
    return postprocess_image(output_tensor)

def main():
    st.set_page_config(
        page_title="Pix2Pix Coloring Bot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Pix2Pix Coloring Bot")
    
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(MODELS.keys()),
        help="Choose which model to use for transformation"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### Model Information
        - **Model A**: Trained on real photo/sketch pairs
        - **Model B**: Trained on anime data
        - **Model C**: Combined training on both datasets
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a sketch or anime image to transform"
        )
        
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, use_column_width=True, caption="Uploaded Image")
    
    with col2:
        st.subheader("Output Image")
        
        if uploaded_file is not None:
            with st.spinner(f"Loading {model_name} model..."):
                generator = load_generator(model_name)
            
            if generator is not None:
                with st.spinner("Transforming image..."):
                    image_tensor = preprocess_image(input_image)
                    output_image = transform_image(generator, image_tensor)
                
                st.image(output_image, use_column_width=True, caption="Transformed Image")
                
                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")
                buffer.seek(0)
                
                st.download_button(
                    label="⬇Download Result",
                    data=buffer.getvalue(),
                    file_name=f"transformed_{model_name.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
        else:
            st.info("Please upload an image")

if __name__ == "__main__":
    main()
