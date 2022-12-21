import glob #returns all file path that match a specific pattern
import pandas as pd

def write_tess_ravdess_csv(emotions=["sad", "neutral", "happy" , "calm", "angry"], train_name="train_tess_ravdess.csv",
                            test_name="test_tess_ravdess.csv", verbose=1):
    
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    for category in emotions:
        # for training speech directory
        total_files = glob.glob(f"data/training/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(category)
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} training audio files for category:{category}")
    
        # for validation speech directory
        total_files = glob.glob(f"data/validation/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(category)
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} testing audio files for category:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)


