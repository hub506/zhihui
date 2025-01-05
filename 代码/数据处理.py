import os   # 用于与操作系统交互，如文件路径操作
import shutil   # 用于文件复制。

dataset_path = "COVID-19_Radiography_Dataset"  # 数据集源路径

output_path = "new_COVID_19_Radiography_Dataset"  # 数据集的产出路径

# 创建test、train和val文件夹
os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val"), exist_ok=True)

# 创建类别文件夹
categories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
for category in categories:
    os.makedirs(os.path.join(output_path, "test", category, "images"),
                exist_ok=True)  # 创建 new_COVID_19_Radiography_Dataset\test\COVID\images
    os.makedirs(os.path.join(output_path, "test", category, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train", category, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train", category, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val", category, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val", category, "masks"), exist_ok=True)

for category in categories:
    image_folder = os.path.join(dataset_path, category, "images")  # COVID-19_Radiography_Dataset\COVID\images
    mask_folder = os.path.join(dataset_path, category, "masks")  # COVID-19_Radiography_Dataset\COVID\masks

    # 获取文件列表
    images = sorted(os.listdir(image_folder))  # 返回images文件夹中所有图像组成的列表
    masks = sorted(os.listdir(mask_folder))


    # 计算划分的索引
    total_images = len(images)  # covid 3616
    test_size = int(0.2 * total_images)  # 3616*0.2
    val_size = int(0.1 * (total_images - test_size))  # 从80% 中取 10% = 8%

    # 划分数据集
    test_images = images[:test_size]  # 723
    val_images = images[test_size:test_size + val_size]
    train_images = images[test_size + val_size:]

    # 复制图片和mask到对应的文件夹
    for img in test_images:
        src_img = os.path.join(image_folder, img)  # COVID-19_Radiography_Dataset\COVID\images
        src_mask = os.path.join(mask_folder, img)
        dest_img = os.path.join(output_path, "test", category, "images",
                                img)  # new_COVID_19_Radiography_Dataset\test\COVID\images
        dest_mask = os.path.join(output_path, "test", category, "masks", img)
        shutil.copy(src_img, dest_img)
        shutil.copy(src_mask, dest_mask)

    for img in val_images:
        src_img = os.path.join(image_folder, img)
        src_mask = os.path.join(mask_folder, img)
        dest_img = os.path.join(output_path, "val", category, "images", img)
        dest_mask = os.path.join(output_path, "val", category, "masks", img)
        shutil.copy(src_img, dest_img)
        shutil.copy(src_mask, dest_mask)

    for img in train_images:
        src_img = os.path.join(image_folder, img)
        src_mask = os.path.join(mask_folder, img)
        dest_img = os.path.join(output_path, "train", category, "images", img)
        dest_mask = os.path.join(output_path, "train", category, "masks", img)
        shutil.copy(src_img, dest_img)
        shutil.copy(src_mask, dest_mask)