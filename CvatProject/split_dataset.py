import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_ratio=0.8):
    """Veri setini train ve test olarak böl"""
    # Kaynak dizinler
    images_dir = os.path.join(source_dir, 'images')
    labels_dir = os.path.join(source_dir, 'labels')
    
    # Hedef dizinler
    train_images_dir = os.path.join(source_dir, 'images', 'train')
    test_images_dir = os.path.join(source_dir, 'images', 'test')
    train_labels_dir = os.path.join(source_dir, 'labels', 'train')
    test_labels_dir = os.path.join(source_dir, 'labels', 'test')
    
    # Dizinleri oluştur
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    # Tüm dosyaları listele
    all_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Dosyaları karıştır
    random.shuffle(all_files)
    
    # Train/test ayrımı
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    # Dosyaları kopyala
    for file in train_files:
        # Görüntü dosyasını kopyala
        shutil.copy2(
            os.path.join(images_dir, file),
            os.path.join(train_images_dir, file)
        )
        # Etiket dosyasını kopyala
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy2(
                os.path.join(labels_dir, label_file),
                os.path.join(train_labels_dir, label_file)
            )
    
    for file in test_files:
        # Görüntü dosyasını kopyala
        shutil.copy2(
            os.path.join(images_dir, file),
            os.path.join(test_images_dir, file)
        )
        # Etiket dosyasını kopyala
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy2(
                os.path.join(labels_dir, label_file),
                os.path.join(test_labels_dir, label_file)
            )
    
    print(f"Veri seti bölündü:")
    print(f"Train seti: {len(train_files)} görüntü")
    print(f"Test seti: {len(test_files)} görüntü")

def main():
    # Veri seti dizini
    dataset_dir = "dataset"
    
    # Veri setini böl
    split_dataset(dataset_dir)

if __name__ == "__main__":
    main() 