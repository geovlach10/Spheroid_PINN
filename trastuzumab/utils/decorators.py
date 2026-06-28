import time

def auto_save(func):
    def wrapper(self, epochs, *args, **kwargs):
        # Pre-excecution 
        print(f"\n🚀 [STAGE START] Running: {func.__name__}")
        print(f"🎯 Target: {epochs} epochs")
    
        start_time = time.time()

        # Calling
        result = func(self, epochs, *args, **kwargs)

        # Post-excecution (experiment duration and saving)
        duration = time.time() - start_time
        print(f"✅ [STAGE DONE] Completed in {duration:.2f}s")

        self.save_checkpoint(stage_name=func.__name__) if hasattr(self, 'save_checkpoint') else print(f"⚠️ Warning: Solver missing 'save_checkpoint' method. Data not saved!")
        return result
    return wrapper