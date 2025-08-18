import torch
import os
import sys
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
try:
    import decord
except ImportError:
    print("ERROR: decord is not installed. Please ensure your CBAS virtual environment is activated.")
    sys.exit(1)

def get_attention_map_for_model(model_identifier: str, image: Image, device: torch.device):
    """
    Loads a specified DINO model, computes the attention map for a given image,
    and returns it as a PIL Image.
    """
    print(f"\n--- Processing model: {model_identifier} ---")
    try:
        config = AutoConfig.from_pretrained(model_identifier)
        config.output_attentions = True
        
        processor = AutoImageProcessor.from_pretrained(model_identifier)
        model = AutoModel.from_pretrained(model_identifier, config=config).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model. {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1]

    # =========================================================================
    # START OF FINAL, CORRECTED LOGIC
    # =========================================================================
    # Get all parameters directly from the model's configuration.
    num_register_tokens = getattr(model.config, 'num_register_tokens', 0)
    num_special_tokens = 1 + num_register_tokens

    # We want the attention from the [CLS] token to the patch tokens
    cls_attentions = attentions[0, :, 0, :]
    patch_attentions = cls_attentions[:, num_special_tokens:].mean(dim=0)
    
    # The number of patches is the length of this tensor. This is our ground truth.
    num_patches = patch_attentions.shape[0]
    
    # The feature map shape is a square root of the number of patches.
    # We assume a square grid, which is standard for ViT models.
    side_length = int(np.sqrt(num_patches))
    if side_length * side_length != num_patches:
        print(f"ERROR: The number of patches ({num_patches}) is not a perfect square. Cannot reshape to a square grid.")
        return None
        
    h_featmap = side_length
    w_featmap = side_length
    
    print(f"Ground Truth from Model Output:")
    print(f"  - Register Tokens: {num_register_tokens}")
    print(f"  - Actual Patch Count: {num_patches}")
    print(f"  - Inferred Patch Grid: {h_featmap}h x {w_featmap}w")
    # =========================================================================
    # END OF FINAL, CORRECTED LOGIC
    # =========================================================================
        
    attention_map = patch_attentions.reshape(h_featmap, w_featmap)
    
    attention_map = attention_map.cpu().numpy()
    min_val, max_val = np.min(attention_map), np.max(attention_map)
    if max_val > min_val:
        attention_map = (attention_map - min_val) / (max_val - min_val)
    
    resized_map = Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    
    print("Attention map generated.")
    return resized_map

if __name__ == "__main__":
    print("DINO Encoder Comparison Tool")
    
    video_file = input("Please enter the FULL path to the video file you want to inspect:\n> ").strip()
    frame_idx = int(input("Enter the frame number to visualize (e.g., 5000):\n> ").strip())

    try:
        print(f"\nLoading frame {frame_idx} from video...")
        vr = decord.VideoReader(video_file, ctx=decord.cpu(0))
        if frame_idx >= len(vr):
            raise ValueError(f"Frame number {frame_idx} is out of bounds.")
        
        frame = vr[frame_idx].asnumpy()
        original_image = Image.fromarray(frame)
        print("Frame loaded.")
    except Exception as e:
        print(f"ERROR: Failed to load video frame: {e}")
        sys.exit(1)

    models_to_compare = {
        "DINOv2 (Base)": "facebook/dinov2-base",
        "DINOv2 (w/ Registers)": "facebook/dinov2-with-registers-base",
        "DINOv3": "facebook/dinov3-vitb16-pretrain-lvd1689m"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle(f'Encoder Comparison\n{os.path.basename(video_file)} - Frame {frame_idx}', fontsize=16)

    for i, (model_display_name, model_id) in enumerate(models_to_compare.items()):
        attention_image = get_attention_map_for_model(model_id, original_image, device)
        
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"Input for:\n{model_display_name}")
        axes[i, 0].axis('off')
        
        if attention_image:
            axes[i, 1].imshow(attention_image, cmap='viridis')
            axes[i, 1].set_title("CLS Token Attention")
        else:
            axes[i, 1].text(0.5, 0.5, 'Failed to generate map', ha='center', va='center')
        axes[i, 1].axis('off')

    output_filename = "encoder_comparison.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"\n[SUCCESS] Comparison saved to: {output_filename}")
    plt.close(fig)

    print("\nComparison complete.")