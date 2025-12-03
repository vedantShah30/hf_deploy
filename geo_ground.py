def load_geoground_pipeline():
    """
    Loads the GeoGround pipeline with proper error handling.
    """
    print("🔧 Loading GeoGround Pipeline...")

    import os
    import sys
    
    # Add the src folder to Python path
    current_dir = os.getcwd()
    possible_paths = [
        os.path.join(current_dir, 'Final_Grounding', 'src'),
        os.path.join(current_dir, 'src'),
        os.path.join(current_dir, 'tati', 'Final_Grounding', 'src'),  # Based on your paths
        '/content/Final_Grounding/src',  # Absolute Colab path
        '/content/tati/Final_Grounding/src'  # Another possibility
    ]
    
    src_path = None
    for path in possible_paths:
        print(f"   Checking: {path}")
        if os.path.exists(path):
            src_path = path
            print(f"   ✅ Found!")
            break
    
    if src_path is None:
        print(f"❌ 'src' folder not found in any of these locations:")
        # for p in possible_paths:
        #     print(f"   - {p}")
        # print("\n💡 Please verify your folder structure and update the paths.")
        return None
    parent_folder = os.path.dirname(src_path)

    if parent_folder not in sys.path:
        sys.path.insert(0, parent_folder)
        print(f"✅ Added PARENT to sys.path: {parent_folder}")
    
    try:
        # Import the pipeline
        from src.pipeline import YOLOQwenPipeline
        
        # ===== CRITICAL: Provide the required paths =====
        # Update these paths according to your setup
        yolo_base_path = "/content/tati/Final_Grounding/yolo11n-obb.pt"  # Base YOLO model
        lora_weights_path = "/content/tati/Final_Grounding/lora_weights.pt"  # Your LoRA adapter weights
        qwen_model_name = "Qwen/Qwen2-VL-7B-Instruct"  # Qwen model
        
        # Initialize with required parameters
        pipeline = YOLOQwenPipeline(
            yolo_base_path=yolo_base_path,
            lora_weights_path=lora_weights_path,
            qwen_model_name=qwen_model_name
        )
        
        print("✅ GeoGround Pipeline Loaded Successfully")
        return pipeline
        
    except ImportError as e:
      print(f"❌ Failed to import GeoGround: {e}")
      print(f"   sys.path: {sys.path}")
        
        # Debug: Check what files exist
      print(f"\n🔍 Files in src folder:")
      try:
          files = os.listdir(src_path)
          for f in files:
              print(f"      - {f}")
      except Exception as ex:
          print(f"   Error listing: {ex}")
        
      return None
    except Exception as e:
        print(f"❌ Error initializing GeoGround: {e}")
        import traceback
        print(traceback.format_exc())
        return None



def run_grounding(model_zoo, image_path, query, conf_threshold=0.3):
    """
    Executes the GeoGround pipeline to find objects.
    Returns formatted text response with bounding boxes.
    """
    print(f"🎯 Grounding Task: '{query}' on image: {image_path}")

    # 1. Check if pipeline is loaded
    pipeline = model_zoo.geoground  # ← Fixed property name
    if not pipeline:
        return "⚠ Grounding model not loaded. Check 'src.pipeline' imports."

    # 2. Validate image exists
    import os
    if not os.path.exists(image_path):
        return f"❌ Image not found: {image_path}"

    # 3. Run the GeoGround pipeline
    try:
        # The new pipeline.process() returns a dict with specific keys
        result = pipeline.process(
            image_path=image_path,
            query=query,
            conf_threshold=conf_threshold  # Optional: adjust detection confidence
        )
        
        # 4. Parse and format results from the new structure
        final_coords = result.get('final_coordinates', [])
        yolo_coords = result.get('yolo_coordinates', [])
        num_detections = result.get('num_detections', 0)
        
        # Format output nicely
        if num_detections == 0:
            return f"No objects found matching '{query}'."
        
        elif num_detections == 1:
            box = final_coords[0]
            return f"Found 1 instance of '{query}' at bounding box: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}]"
        
        else:
            box_strings = []
            for i, box in enumerate(final_coords, 1):
                box_strings.append(
                    f"  Object {i}: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}]"
                )
            
            response = f"Found {num_detections} instances of '{query}':\n"
            response += "\n".join(box_strings)
            response += f"\n\n(YOLO initially detected {len(yolo_coords)} candidates, refined by Qwen)"
            
            return response
            
    except Exception as e:
        import traceback
        print(f"❌ GeoGround Error: {e}")
        print(traceback.format_exc())
        return f"Error executing grounding: {str(e)}"
