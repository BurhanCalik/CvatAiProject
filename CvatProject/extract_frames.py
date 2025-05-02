import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, target_size=(640, 640)):
    """Video dosyasından frame'leri çıkar"""
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    
    # Video özelliklerini al
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # FPS'i manuel olarak 30 olarak ayarla
    fps = 30
    
    print(f"Video bilgileri:")
    print(f"Toplam frame sayısı: {total_frames}")
    print(f"FPS: {fps}")
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    
    # İlerleme çubuğu
    pbar = tqdm(total=total_frames, desc="Frame'ler çıkarılıyor")
    
    while True:
        # Frame'i oku
        ret, frame = cap.read()
        
        # Video bittiyse döngüden çık
        if not ret:
            break
        
        # Frame'i yeniden boyutlandır
        frame = cv2.resize(frame, target_size)
        
        # Frame'i kaydet (JPEG kalitesini düşür)
        frame_path = os.path.join(output_dir, f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        frame_count += 1
        pbar.update(1)
    
    # Video dosyasını kapat
    cap.release()
    pbar.close()
    
    print(f"\nToplam {frame_count} frame kaydedildi.")
    print(f"Frame'ler {output_dir} klasörüne kaydedildi.")
    print(f"Her frame {target_size[0]}x{target_size[1]} boyutunda ve %85 JPEG kalitesinde kaydedildi.")

def main():
    # Video dosyasının yolu
    video_path = "input_video.mov"  # .mov uzantılı video
    
    # Çıktı dizini
    output_dir = "dataset/images"
    
    # Frame'leri çıkar
    extract_frames(
        video_path, 
        output_dir,
        target_size=(640, 640)  # YOLO için standart boyut
    )

if __name__ == "__main__":
    main() 