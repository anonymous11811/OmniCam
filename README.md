# OmniCam  
## Configuration Instructions  
1. **Run ViewCrafter first**  
2. **Download weights**  
   ```bash  
   wget https://drive.google.com/drive/folders/1FUqwUt95QCL0mblZOQeXEJdGAmXRPXX5?usp=sharing  
   ```  
3. **Expected directory structure (with sub-branch in llamacheckpoint)**  
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
