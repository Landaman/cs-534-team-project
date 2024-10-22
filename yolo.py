from ultralytics import YOLO

def train_yolo(dataset_yaml: str, epochs: int, use_pretrained: bool) -> None:
    if use_pretrained:
        # Load a pretrained YOLO model (recommended for training)
        model = YOLO("yolo11n.pt")
    else:
        # Create a new YOLO model from scratch
        model = YOLO("yolo11n.yaml")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    model.train(data=dataset_yaml, epochs=epochs)

    # Evaluate the model's performance on the validation set
    model.val()

    # Export the model to ONNX format
    model.export(format="onnx")

if __name__ == '__main__':
    train_yolo("./datasets/indoor-objects-detection-10-categories/data.yaml", 3, True)
