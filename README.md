# OmniCam  
OmniCam is an image motion video generation system that automatically converts user input images and natural language descriptions (such as "camera moves left and then rotates up") into videos.The specific workflow is as follows: Firstly, the LLaMA big language model is used to parse the text description into structured camera motion parameters (including time, speed, direction, etc.), then these parameters are converted into specific camera trajectory coordinates, and finally, the ViewCrafter model is used to generate the final video based on input images and trajectory data. The entire system provides a complete processing pipeline and web interface, allowing users to generate video content with specific camera motion effects through simple text descriptions.
## Configuration Instructions  
1. **Try to run ViewCrafter first**  
2. **Download weights**  
   ```bash  
   wget https://drive.google.com/drive/folders/1FUqwUt95QCL0mblZOQeXEJdGAmXRPXX5?usp=sharing  
   ```  
3. **Expected directory structure**  
   ```markdown  
   ├── Viewcrafter  
   ├── weight  
   │   └── llamacheckpoint  
   │       └── best_model2.pth 
   │           ├── adapter_config.json
   │           ├── adapter_model.safetensors 
   │           └── README.md  
   └── description2video  
       ├── description-descriptionjson.py  
       ├── descriptionjson-trajectoryjson.py  
       ├── trajectoryjson-viewcrafterinput.py  
       ├── viewcrafterinput-video.py  
       ├── pipeline.py  
       ├── inferenceinput1.txt  
       └── gradio.py  
   ```  
4. **Notice**  
   - Ensure the sub-branch structure under `llamacheckpoint` matches the model requirements.  
   - Compatibility issues may occur with newer Gradio versions. Use the specified version or check official updates.
# Description2Video Project

This is an AI project for generating videos based on text descriptions, converting natural language descriptions into camera trajectories and ultimately generating videos through a multi-step processing pipeline.

## Project Overview

This project implements a complete text-to-video generation pipeline, including the following main steps:
1. **Text Description** → **Structured JSON** (using LLaMA model)
2. **Structured JSON** → **Trajectory JSON** (camera motion parameters)
3. **Trajectory JSON** → **ViewCrafter Input Format** (text file)
4. **ViewCrafter Input** → **Final Video** (using ViewCrafter model)

## File Description

### Core Processing Files

#### `pipeline.py` - Main Processing Pipeline
- **Function**: Complete end-to-end processing workflow, including all steps
- **Main Components**:
  - `generate_description_json()`: Uses LLaMA model to convert text descriptions to structured JSON
  - `Text2VideoSet`: Core class for converting description JSON to trajectory JSON
  - `convert_trajectory_to_txt()`: Converts trajectory JSON to ViewCrafter input format
  - `run_viewcrafter_inference()`: Calls ViewCrafter to generate final video
- **Features**: Includes CUDA/cuDNN configuration and error handling mechanisms

#### `pipegradio_demo.py` - Gradio Web Interface
- **Function**: Provides user-friendly web interface for text-to-video conversion
- **Features**: 
  - Interactive interface built with Gradio
  - Supports text input and image upload
  - Real-time processing and result display

### Single-Step Processing Files

#### `description-descriptionjson.py` - Step 1: Text→Structured JSON
- **Function**: Converts natural language descriptions to structured JSON containing time, speed, direction and other information
- **Input**: Text description file (`inferenceinput.txt`)
- **Output**: Structured JSON file
- **Model**: Uses fine-tuned LLaMA-3.1-8B model

#### `descriptionjson-trajectoryjson.py` - Step 2: Structured JSON→Trajectory JSON
- **Function**: Converts structured description information to specific camera trajectory parameters
- **Core Class**: `Text2VideoSet`
- **Processing Logic**:
  - `process_direction()`: Converts direction and speed to specific angle changes
  - `tune_pose()`: Generates complete camera pose sequences
- **Output**: JSON file containing phi, theta, r coordinate sequences

#### `trajectoryjson-viewcrafterinput.py` - Step 3: Trajectory JSON→ViewCrafter Input
- **Function**: Converts trajectory JSON to text format required by ViewCrafter model
- **Input**: Trajectory JSON file
- **Output**: Three-line text file containing theta, phi, r sequences respectively
- **Format**: Space-separated numerical sequences per line

#### `viewcrafterinput-video.py` - Step 4: Generate Final Video
- **Function**: Calls ViewCrafter model to generate final video
- **Input**: 
  - Input image (PNG format)
  - Trajectory text file
- **Output**: Generated video file
- **Parameters**: Contains detailed ViewCrafter inference parameter configuration

### Configuration Files

#### `requirements.txt` - Dependency Package List
- **Function**: Contains all Python dependency packages required by the project
- **Main Dependencies**:
  - PyTorch related packages
  - Transformers (Hugging Face)
  - Gradio (Web interface)
  - Other data processing and visualization packages

#### `inferenceinput.txt` - Sample Input File
- **Function**: Contains text description examples for testing
- **Format**: Natural language description of camera movements

## Usage
```bash
cd description2video
```
### Method 1: Use Complete Pipeline (Recommended)
```bash
python pipeline.py
```

### Method 2: Use Web Interface
```bash
python pipegradio_demo.py
```
Then open the displayed local address in your browser.

### Method 3: Step-by-Step Execution
```bash
# Step 1: Text to Structured JSON
python description-descriptionjson.py

# Step 2: Structured JSON to Trajectory JSON  
python descriptionjson-trajectoryjson.py

# Step 3: Trajectory JSON to ViewCrafter Input
python trajectoryjson-viewcrafterinput.py

# Step 4: Generate Video
python viewcrafterinput-video.py
```

## Environment Requirements

- Python 3.9
- CUDA-supported GPU (recommended)
- Sufficient VRAM (40GB+ recommended)

## Install Dependencies

```bash
cd description2video
pip install -r requirements.txt
```

## Important Notes

1. **Model Paths**: Need to modify model paths in the code according to actual situation
2. **CUDA Configuration**: Project is optimized for CUDA environment, CPU mode has lower performance
3. **Memory Management**: Large model inference requires significant memory, recommend monitoring resource usage
4. **File Paths**: Please check and modify file path configurations in the code before running

## Output Formats

- **Intermediate Files**: JSON format structured data
- **Final Output**: MP4 format video files
- **Trajectory Data**: Text files containing camera motion parameters

## Technology Stack

- **Deep Learning Framework**: PyTorch
- **Natural Language Processing**: Transformers, LLaMA
- **Video Generation**: ViewCrafter
- **Web Interface**: Gradio
- **Data Processing**: NumPy, Pandas 
