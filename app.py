import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import numpy as np
import cv2

def image_to_bytes(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()

st.set_page_config(page_title="Latent Doodles Sandbox", layout="centered")
st.title(" Latent Doodles – Interactive Art Sandbox (Local)")
st.write("Draw a doodle and watch AI turn it into art. All runs locally, no API!")

@st.cache_resource
def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe

pipe = load_pipeline()

st.markdown("**Draw your doodle below:**")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=8,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=384,
    height=384,
    drawing_mode="freedraw",
    key="canvas"
)

prompt = st.text_input("Prompt (describe what you want, e.g., 'a cat astronaut in Van Gogh style')", "a masterpiece painting, Van Gogh style")
seed = st.number_input("Random seed (0=random)", value=0, min_value=0, step=1)

if st.button("Generate Artwork"):
    if canvas_result.image_data is not None:
        # Prepare the doodle
        np_img = (canvas_result.image_data[..., :3] * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img).convert("RGB")

        # Convert to 'scribble' format (binarize edges)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        scribble = np.stack([edges]*3, axis=-1)
        pil_scribble = Image.fromarray(scribble)

        st.image(pil_scribble, caption="Edge/Scribble Input", width=256)

        generator = None
        if seed > 0:
            generator = torch.manual_seed(seed)

        with st.spinner("Generating... (may take ~10-20s)"):
            output = pipe(
                prompt,
                image=pil_scribble,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=8.5,
            )
            image = output.images[0]
            st.image(image, caption="AI Art", width=384)
            st.success("Done! Download or share your creation.")
            st.download_button("Download", data=image_to_bytes(image), file_name="ai_art.png", mime="image/png")
    else:
        st.warning("Draw something first!")

