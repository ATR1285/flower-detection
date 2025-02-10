

### Project Title  
Flower Identification AI  

### Description
This project is a deep learning-based AI model designed to classify flowers into three categories: **daisy, rose, and tulip**. The model uses a dataset of images stored in structured folders and is trained using PyTorch. It aims to identify flower types and estimate their lifespan using a mobile camera.

### Dataset Structure
The dataset is organized into three main folders:  
- train/ → Contains images for training the model  
- test/ → Contains images for testing the model  
- validation/ → Contains images for model validation  

Each folder contains subfolders for the three flower categories:  
- `daisy/`  
- `rose/`  
- `tulip/`  

### Setup Instructions
1. Ensure Dataset Availability  
   - The dataset should be placed inside the `dataset/` directory.  
   - The expected folder structure is:  
     ```
     dataset/
       ├── train/
       │   ├── daisy/
       │   ├── rose/
       │   ├── tulip/
       ├── test/
       │   ├── daisy/
       │   ├── rose/
       │   ├── tulip/
       ├── validation/
           ├── daisy/
           ├── rose/
           ├── tulip/
     ```
   - Make sure that each flower type folder contains images.

2. Verify Folder and Image Availability
   - The code includes checks to ensure that the required dataset directories exist.  
   - If folders appear empty, confirm that images are correctly placed inside the respective subfolders.  

3. Image Preprocessing & Renaming
   - The images inside the folders should follow a consistent naming format (e.g., `image_1.jpg`, `image_2.jpg`, etc.).  
   - Previously, images were named like `img 1(1), img 1(2)`, etc., and have been standardized for proper loading.  
   - A renaming script has been implemented to ensure consistency.  

4. Dependencies  
   - The project runs in a Python environment with the following libraries:  
     - PyTorch  
     - torchvision  
     - NumPy  
     - OpenCV (if additional image processing is required)  

5. Training the Model
   - The model uses PyTorch and runs on a CPU environment.  
   - It is built using a CNN-based architecture and is trained using cross-entropy loss.  
   - The Adam optimizer is used with a learning rate of `0.001`.  

### Troubleshooting
- If an error occurs stating "Folder not found", ensure that the dataset structure is correct.  
- If subfolders inside `dataset/train` are empty, confirm that the images exist in the appropriate subdirectories.  
- If images are not loading, check their extensions (must be `.jpg`, `.png`, etc.).  

### Fture Improvements  
- Add support for additional flower categories.  
- Optimize the model for better accuracy.  
- Implement a real-time mobile app interface for flower detection.  

