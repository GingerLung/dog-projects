from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 Stable Diffusion 模型
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to(device)

# 設定生成參數
output_base_dir = r"C:\Users\123\桌面\doge\angry-clean"
inspection_base_dir = r"C:\Users\123\桌面\doge\angry-clean"

categories = {
    "angry": 200,
    "happy": 50,
    "relaxed":200,
}

# 為每種情緒設定描述（prompts）
prompts = {
    "angry": [
        "a fierce and angry dog showing teeth, realistic, 4k, high quality",
        "a dog with stiff body posture, ears back, and eyes glaring, realistic, 4k, high quality",
        "an aggressive dog with raised hackles, tail held high, and a rigid stance, realistic, 4k, high quality",
        "a growling dog with narrowed eyes, ears pinned back, and tense muscles, realistic, 4k, high quality"
    ],
    "happy": [
        "a joyful dog wagging its tail, mouth open, friendly expression, realistic, 4k, high quality",
        "a playful dog running on grass, tongue out, and ears perked up, realistic, 4k, high quality",
        "a cheerful dog rolling on its back, looking content and relaxed, realistic, 4k, high quality",
        "a happy dog jumping with excitement, tail wagging, and a bright expression, realistic, 4k, high quality"
    ],
    "relaxed": [
        "a relaxed dog lying down peacefully, calm expression, realistic, 4k, high quality",
        "a calm dog sitting in a sunny garden, tail resting on the ground, realistic, 4k, high quality",
        "a content dog sleeping on a couch, ears slightly drooped, realistic, 4k, high quality",
        "a relaxed dog stretching gently, looking at the horizon with a soft expression, realistic, 4k, high quality"
    ],
    "sad": [
        "a sad dog with a drooping tail, ears down, melancholic expression, realistic, 4k, high quality",
        "a lonely dog sitting in the rain, looking forlorn, realistic, 4k, high quality",
        "a dejected dog curled up on the floor, eyes looking downward, realistic, 4k, high quality",
        "a grieving dog with low ears, tail tucked, and a sorrowful gaze, realistic, 4k, high quality"
    ]
}

# 讀取資料集中圖片的像素大小
def get_dataset_image_size(dataset_dir):
    sizes = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    sizes.append(img.size)  # (width, height)
    if sizes:
        # 統計圖片尺寸有哪些
        most_common_size = max(set(sizes), key=sizes.count)
        print(f"資料集中最常見的圖片大小為：{most_common_size}")
        return most_common_size
    else:
        raise ValueError("資料集中未找到有效的圖片檔案！")

# 獲得資料夾中現有圖片的最大編號
def get_start_index(output_dir, category):
    files = os.listdir(output_dir)
    indices = [
        int(f.split("_")[1].split(".")[0])  # 提取數字部分
        for f in files if f.startswith(category) and f.endswith(".jpg")
    ]
    return max(indices, default=0) + 1  # 如果沒有檔案則從 1 開始

# 生成圖片並且調整大小
def generate_images(category, num_images, prompts, output_dir, inspection_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(inspection_dir, exist_ok=True)
    start_index = get_start_index(inspection_dir, category)  # 獲取起始編號
    print(f"開始生成 {category} 圖片，從編號 {start_index} 開始...")
    for i in range(num_images):
        # 隨機選擇一個 prompt
        prompt = random.choice(prompts)
        image = pipe(prompt).images[0]
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        # 保存圖片到 generated_inspection 目錄
        current_index = start_index + i
        inspection_path = os.path.join(inspection_dir, f"{category}_{current_index:04d}.jpg")
        image.save(inspection_path)
        print(f"已保存檢查圖片：{inspection_path}，大小為：{image.size}")

# 主程式
dataset_dir = r"C:\Users\123\桌面\doge\angry-clean"
target_size = get_dataset_image_size(dataset_dir)  # 獲得資料集中圖片的大小

for category, num_images in categories.items():
    output_dir = os.path.join(output_base_dir, category)
    inspection_dir = os.path.join(inspection_base_dir, category)
    generate_images(category, num_images, prompts[category], output_dir, inspection_dir, target_size)

print("圖片生成完成且已統一大小！")