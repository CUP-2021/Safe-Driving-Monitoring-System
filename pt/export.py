from ultralytics import YOLO

# Load a model
model = YOLO('633 and 0.98best.pt')  # load a custom trained model

# Export the model
model.export(format='engine',device=0)
