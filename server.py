from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Enable CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dynamic Analysis Logic ---
def analyze_freshness(image):
    """
    Estimates freshness based on color saturation and brightness.
    Bright, saturated colors = Fresh.
    Dull, dark colors = Older/Wilted.
    """
    # Convert to HSV
    hsv_img = image.convert("HSV")
    # Get stats
    from PIL import ImageStat
    stat = ImageStat.Stat(hsv_img)
    # stat.mean returns [H, S, V] mean values
    # Saturation is index 1 (0-255)
    mean_saturation = stat.mean[1]
    mean_brightness = stat.mean[2]
    
    # Normalize (0-1)
    sat_score = mean_saturation / 255.0
    val_score = mean_brightness / 255.0
    
    # Freshness Score (0.0 to 1.0) weighted 70% Saturation, 30% Brightness
    freshness = (sat_score * 0.7) + (val_score * 0.3)
    
    # Map to days (Max ~10 days)
    days_left = int(freshness * 10)
    
    if days_left >= 7:
        status = "Very Fresh"
        desc = "Vibrant colors indicate it was likely cut recently."
    elif days_left >= 4:
        status = "Peak Bloom"
        desc = "Standard condition. Keep watered."
    elif days_left >= 2:
        status = "Aging"
        desc = "Colors are fading. Change water immediately."
    else:
        status = "Wilting"
        desc = "End of life cycle."
        
    return f"{str(max(1, days_left))} Days ({status})", desc

# --- Data Schemas ---
# We keep generic meanings, but use specific analysis for lifespan
FLOWER_INFO = {
    "daisy": {"meaning": "Purity, Innocence, and Loyal Love", "care": "Remove leaves below water."},
    "rose": {"meaning": "Love and Passion", "care": "Trim stems at an angle."},
    "tulip": {"meaning": "Perfect and Deep Love", "care": "Keep in cold water."}
}

# --- Load Personal Model (ResNet18) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_flower_model.pth"

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    
    model = models.resnet18(weights=None) # No need for pretrained weights as we load state_dict
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded Personal ML Model: {MODEL_PATH}")
    print(f"Classes: {classes}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    classes = [] # Fail safe

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            score = confidence.item()
            idx = predicted.item()
            detected_flower = classes[idx]
        
        # Threshold Logic (Personal ML might be 99% confident on wrong things if OOD)
        # But user requested "Personal ML", so we trust it mostly.
        # We can keep a small threshold.
        if score > 0.4: # Relaxed threshold further
            
            # Run Dynamic Analysis
            lifespan_est, condition_desc = analyze_freshness(image)
            base_info = FLOWER_INFO.get(detected_flower, {"meaning": "Beautiful Flower", "care": "Water regularly."})
            
            return {
                "success": True,
                "flower": detected_flower,
                "confidence": round(score * 100, 2),
                "data": {
                    "lifespan": lifespan_est,
                    "meaning": base_info["meaning"],
                    "care": f"{base_info['care']} Note: {condition_desc}"
                }
            }
        else:
            return {
                "success": False,
                "message": f"Low confidence ({round(score*100, 1)}%). Try a clearer photo."
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

# Serve static files (PWA)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
