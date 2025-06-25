import os
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import load
'''
model_name = "ViT-L/14@336px"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入模型與預處理方法
model, preprocess = load(model_name, device=device)
model = model.float()  # 確保模型使用 FP32

# 處理圖片的轉換
transform = Compose([
    Resize((model.visual.input_resolution, model.visual.input_resolution)),
    CenterCrop(model.visual.input_resolution),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device).float()  # 確保輸入數據為 FP32
    return image

def extract_features(image_tensor):
    with torch.no_grad():
        # 模型輸出 class token 和 patch tokens
        features = model.visual(image_tensor)
        class_token = features[0]
        patch_tokens = features[1]
    return class_token.cpu(), patch_tokens.cpu()

coco_image_dir = "/DATA3/flaya/iFSD/data/coco/val2017"
output_dir = "/DATA3/flaya/iFSD/myclip/out_val"
# 遍歷 COCO 圖片並萃取特徵
image_files = [f for f in os.listdir(coco_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files = sorted(image_files)

for image_file in image_files:
    print(image_file)
    image_path = os.path.join(coco_image_dir, image_file)
    image_tensor = process_image(image_path)
    
    class_token, patch_tokens = extract_features(image_tensor)

    # 儲存特徵
    torch.save(class_token, os.path.join(output_dir, f"{image_file}_class_token.pt"))
    torch.save(patch_tokens, os.path.join(output_dir, f"{image_file}_patch_tokens.pt"))

print(f"完成！特徵已儲存到 {output_dir}")'''
file_name = "/DATA3/flaya/iFSD/myclip/out_val/000000000139.jpg_class_token.pt"
state_dict = torch.load(file_name)

print(state_dict.shape)