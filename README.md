# Flower AI Pro - Web Application

A beautiful web-based flower identification system using your Personal ML model (ResNet18).

## 🚀 Quick Start

### Run the Server
```bash
python server.py
```

### Access the Website

**On your computer**:
```
http://localhost:8000
```

**On your phone/tablet** (same WiFi):
```
http://192.168.60.155:8000
```
*(Replace with your actual IP from `ipconfig`)*

## 📱 Features

- ✅ **Works on all devices** (phone, tablet, computer)
- ✅ **Camera access** on mobile devices
- ✅ **Personal ML model** (your trained ResNet18)
- ✅ **Dynamic freshness analysis** (color-based lifespan estimation)
- ✅ **Beautiful responsive UI** with gradient design
- ✅ **PWA capable** (can be installed on home screen)

## 🌐 Make It Public (Optional)

### Using ngrok (Easiest)
```bash
# Download from https://ngrok.com/download
ngrok http 8000
```
Share the generated `https://` link with anyone!

### Using Cloudflare Tunnel
```bash
# Download from https://developers.cloudflare.com/cloudflare-one/
cloudflared tunnel --url http://localhost:8000
```

## 📂 Project Structure

```
Flower-Detection/
├── server.py                 # FastAPI backend with ML model
├── best_flower_model.pth     # Your trained ResNet18 model
├── static/
│   ├── index.html           # Web app UI
│   ├── sw.js                # Service worker (PWA)
│   └── manifest.json        # App manifest
├── train.py                 # Model training script
├── predict.py               # CLI prediction tool
└── data_temp/               # Training data (daisy, rose, tulip)
```

## 🎨 Supported Flowers

- 🌼 **Daisy**: Purity, Innocence, and Loyal Love
- 🌹 **Rose**: Love and Passion
- 🌷 **Tulip**: Perfect and Deep Love

## 🔧 How It Works

1. **User uploads/captures** flower image
2. **Browser sends** image to FastAPI server
3. **Server runs** ResNet18 inference
4. **Analyzes freshness** using HSV color analysis
5. **Returns** flower type, confidence, lifespan, care tips

## 📊 Technical Details

- **Backend**: FastAPI + PyTorch
- **Model**: ResNet18 (44.8 MB)
- **Frontend**: Pure HTML/CSS/JavaScript
- **Freshness Algorithm**: Saturation + Brightness analysis
- **Threshold**: 40% confidence minimum

## 🎯 Next Steps

1. **Test locally**: Run `python server.py` and visit `http://localhost:8000`
2. **Test on phone**: Use your IP address on same WiFi
3. **Deploy publicly**: Use ngrok or cloud hosting

---

**Status**: ✅ Production Ready • Works on All Devices
