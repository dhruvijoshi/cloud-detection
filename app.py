import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import segmentation_models_pytorch as smp

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

device = torch.device(DEVICE)
print(f"Using device: {device}")


# Load model
model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=4,
    classes=2
)

model.load_state_dict(torch.load("models/cloud_detector.pth", map_location='mps'))
model = model.to("mps")

model.eval()


# Preprocessing
EXPECTED_CHANNELS = 4
INPUT_SIZE = (256, 256)

def preprocess_image(img, expected_channels=EXPECTED_CHANNELS, size=INPUT_SIZE):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("RGBA")
    img = img.resize(size)
    img_array = np.array(img)

    # adjust channel count dynamically
    if img_array.shape[2] > expected_channels:
        img_array = img_array[:, :, :expected_channels]
    elif img_array.shape[2] < expected_channels:
        extra = np.zeros(
            (*img_array.shape[:2], expected_channels - img_array.shape[2]),
            dtype=img_array.dtype
        )
        img_array = np.concatenate([img_array, extra], axis=2)

    tensor = transforms.ToTensor()(img_array).unsqueeze(0).to(device)
    return tensor



# Prediction
def predict(image, threshold=0.5):
    try:
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)

            # handle binary vs multiclass
            if output.shape[1] > 1:
                probs = torch.softmax(output, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            else:
                probs = torch.sigmoid(output)
                pred_mask = (probs > threshold).squeeze().cpu().numpy().astype(np.uint8)

        # create a pure mask image (white clouds, black background)
        mask_img = Image.fromarray(pred_mask * 255).convert("L")

        return mask_img

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise gr.Error(f"Error processing image: {str(e)}")



# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️ Cloud Detection App")
    gr.Markdown("Upload a satellite image — the model will highlight cloud regions.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                  label="Confidence Threshold")
            submit_btn = gr.Button("Detect Clouds", variant="primary")

        output_image = gr.Image(label="Predicted Cloud Mask", type="pil")

    gr.Markdown("---")
    gr.Markdown("""
    ### How to Use:
    1. Upload a satellite image.
    2. Adjust the threshold if needed.
    3. Click 'Detect Clouds' to see the mask.

    **White areas = detected clouds**  
    **Black areas = clear sky**
    """)

    submit_btn.click(
        fn=predict,
        inputs=[input_image, threshold],
        outputs=output_image
    )


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    print("Starting Gradio app...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_api=False,
    )
