"""
从 HuggingFace 镜像下载 CIFAR-100 Parquet 文件，提取 5 类家具
"""
import os, sys, shutil, random, time, io
import numpy as np
from PIL import Image
from collections import defaultdict
import requests
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

FURNITURE_CLASSES = {
    "bed": 5, "chair": 20, "sofa": 25, "table": 84, "cabinet": 94,  # wardrobe->cabinet
}

HF_MIRROR = "https://hf-mirror.com"
PARQUET_FILES = [
    "/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
    "/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
]

random.seed(42)
np.random.seed(42)


def download_file(url, dest):
    """下载文件，带进度"""
    print(f"  下载: {os.path.basename(dest)}")
    r = requests.get(url, stream=True, timeout=120)
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {downloaded/1024/1024:.0f}/{total/1024/1024:.0f}MB "
                          f"({downloaded/total*100:.0f}%)", end="", flush=True)
    print()


def save_image_task(args):
    img, path = args
    img.resize((224, 224), Image.LANCZOS).save(path)


def main():
    print("=" * 60)
    print("从 HF 镜像下载 CIFAR-100 家具数据集")
    print("=" * 60)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parquet_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 下载 Parquet 文件
    print("\n[1/3] 下载 Parquet 文件...")
    parquet_paths = []
    for pf in PARQUET_FILES:
        url = HF_MIRROR + pf
        dest = os.path.join(cache_dir, os.path.basename(pf))
        if not os.path.exists(dest):
            start = time.time()
            download_file(url, dest)
            elapsed = time.time() - start
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"    完成: {size_mb:.0f}MB, 耗时 {elapsed:.0f}s")
        else:
            print(f"  已缓存: {os.path.basename(dest)} ({os.path.getsize(dest)/1024/1024:.0f}MB)")
        parquet_paths.append(dest)

    # 2. 解析 Parquet，提取家具类图片
    print("\n[2/3] 解析数据，提取 5 类家具...")
    all_data = defaultdict(list)

    for pp in parquet_paths:
        table = pq.read_table(pp)
        df = table.to_pandas()
        print(f"  文件: {os.path.basename(pp)}, 行数: {len(df)}")

        for _, row in df.iterrows():
            label = row["fine_label"]
            for cls_name, cls_idx in FURNITURE_CLASSES.items():
                if label == cls_idx:
                    img_data = row["img"]
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                    else:
                        img_bytes = img_data
                    img = Image.open(io.BytesIO(img_bytes))
                    all_data[cls_name].append(img)
                    break

    total = sum(len(v) for v in all_data.values())
    for cls in sorted(all_data):
        print(f"  {cls}: {len(all_data[cls])} 张")
    print(f"  总计: {total} 张")

    # 3. 划分并保存
    print("\n[3/3] 划分 train/val/test (70/15/15) 并保存...")
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)

    save_tasks = []
    for cls_name, images in all_data.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, split_images in splits.items():
            out_dir = os.path.join(SAVE_DIR, split_name, cls_name)
            os.makedirs(out_dir, exist_ok=True)
            for i, pil_img in enumerate(split_images):
                save_tasks.append((pil_img, os.path.join(out_dir, f"{i:04d}.png")))
        print(f"  {cls_name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print(f"\n  保存 {len(save_tasks)} 张图片 (224x224)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(save_image_task, save_tasks))

    # 清理
    shutil.rmtree(cache_dir)

    print(f"\n数据集准备完成！路径: {SAVE_DIR}")
    for split in ["train", "val", "test"]:
        for cls in FURNITURE_CLASSES:
            d = os.path.join(SAVE_DIR, split, cls)
            if os.path.exists(d):
                print(f"  {split}/{cls}/ ({len(os.listdir(d))} 张)")


if __name__ == "__main__":
    main()
