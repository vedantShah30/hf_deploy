from peft import PeftModel

# Global variable to track currently loaded adapter
_current_adapter_name = None
# _loaded_adapters = set()  # Track all loaded adapters

def apply_adapter(model, adapter_path):
    """
    Efficiently switches between LoRA adapters:
    - Reuses already-loaded adapters (just switches)
    - Only loads new adapters when needed
    - Properly tracks state
    """
    import os
    global _current_adapter_name, _loaded_adapters

    if(adapter_path == None and _current_adapter_name != None):
      model.delete_adapter(_current_adapter_name)
      print("Removing adapters and Loading base Model")
      return

    if(adapter_path == None and _current_adapter_name == None):
      # model.set_adapter(None)
      # print("Removing adapters and Loading base Model")
      return

    adapter_name = os.path.basename(adapter_path.rstrip('/'))
    # print(f"🔄 Switching to Adapter: {adapter_name}")


    try:
        # Case 1: Adapter is already active - do nothing
        if _current_adapter_name == adapter_name:
          model.set_adapter(adapter_name)
          print(f"✅ Adapter '{adapter_name}' already active.\n")
          return
        
        # Case 2: Adapter is loaded but not active - just switch

        if(_current_adapter_name != adapter_name and _current_adapter_name == None):
          # model.set_adapter(None)
          model.load_adapter(adapter_path, adapter_name=adapter_name)
          model.set_adapter(adapter_name)
          _current_adapter_name = adapter_name
          print(f"✅ Switched to adapter '{adapter_name}'.\n")
          return
        elif(_current_adapter_name != adapter_name and _current_adapter_name != None):
          model.delete_adapter(_current_adapter_name)
          model.load_adapter(adapter_path, adapter_name=adapter_name)
          model.set_adapter(adapter_name)
          _current_adapter_name = adapter_name
          print(f"✅ Switched to adapter '{adapter_name}'.\n")
          return
        
    except Exception as e:
        print(f"⚠️ Adapter error: {e}")
        print(f"   Continuing with current configuration.\n")
