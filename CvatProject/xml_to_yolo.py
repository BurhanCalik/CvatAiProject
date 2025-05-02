import xml.etree.ElementTree as ET
import os
import numpy as np
from pathlib import Path

def create_bounding_box(points):
    """16 noktadan bounding box oluştur"""
    x_coords = [float(p[0]) for p in points]
    y_coords = [float(p[1]) for p in points]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # YOLO formatı: x_center, y_center, width, height (normalized)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

def convert_keypoints_to_yolo(points, img_width, img_height):
    """Keypoint koordinatlarını YOLO formatına dönüştür"""
    yolo_points = []
    for x, y in points:
        # Normalize coordinates
        x_norm = float(x) / img_width
        y_norm = float(y) / img_height
        yolo_points.extend([x_norm, y_norm])
    return yolo_points

def process_xml(xml_file, output_dir):
    """XML dosyasını işle ve YOLO formatında etiketler oluştur"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Her frame için
    for track in root.findall('.//track'):
        for point in track.findall('points'):
            frame = int(point.get('frame'))
            points_str = point.text.strip().split(';')
            points = [p.split(',') for p in points_str]
            
            # Görüntü boyutlarını al (varsayılan olarak 640x640)
            img_width = 640
            img_height = 640
            
            # Bounding box oluştur
            bbox = create_bounding_box(points)
            
            # Keypoint'leri YOLO formatına dönüştür
            keypoints = convert_keypoints_to_yolo(points, img_width, img_height)
            
            # YOLO formatında satır oluştur
            # Format: class x_center y_center width height kp1_x kp1_y kp2_x kp2_y ...
            yolo_line = f"0 {' '.join(map(str, bbox))} {' '.join(map(str, keypoints))}"
            
            # Dosyaya yaz
            output_file = os.path.join(output_dir, f"{frame:06d}.txt")
            with open(output_file, 'w') as f:
                f.write(yolo_line)

def main():
    # XML dosyasının yolu
    xml_file = "annotations.xml"
    
    # Çıktı dizini
    output_dir = "dataset/labels"
    
    # Dizin yoksa oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # XML'i işle
    process_xml(xml_file, output_dir)
    print("Dönüşüm tamamlandı!")

if __name__ == "__main__":
    main() 