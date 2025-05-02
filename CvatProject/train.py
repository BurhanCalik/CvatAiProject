from ultralytics import YOLO
import yaml

def create_yaml():
    """YOLO için yaml dosyası oluştur"""
    data = {
        'path': 'dataset',  # dataset root dir
        'train': 'images/train',  # train images (relative to 'path')
        'val': 'images/test',  # val images (relative to 'path')
        'names': {
            0: 'mouse'  # class names
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def train_model():
    """YOLOv8 modelini eğit"""
    # YAML dosyasını oluştur
    create_yaml()
    
    # YOLOv8-pose modelini yükle
    model = YOLO('yolov8n-pose.pt')  # küçük model
    
    # Modeli eğit
    results = model.train(
        data='dataset.yaml',
        epochs=5,
        imgsz=640,
        batch=16,
        name='mouse_pose_model'
    )
    
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    train_model() 