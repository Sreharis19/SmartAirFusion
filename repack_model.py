# repack_model.py
import joblib
import __main__

# Import the class from a real module
from detector_runtime import SmartAirFusionDetector

# Make it available as __main__.SmartAirFusionDetector
# so the old pickle can find it during loading.
__main__.SmartAirFusionDetector = SmartAirFusionDetector

OLD = "smartairfusion_detector.joblib"
NEW = "smartairfusion_detector_portable.joblib"

obj = joblib.load(OLD)     # now it can unpickle successfully
joblib.dump(obj, NEW)      # new pickle will reference detector_runtime.SmartAirFusionDetector

print("Repacked model saved as:", NEW)
