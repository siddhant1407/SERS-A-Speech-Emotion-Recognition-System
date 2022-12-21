from data_extractor import load_data
from utils import AVAILABLE_EMOTIONS
from create_csv import write_tess_ravdess_csv
from utils import get_audio_config

import os

class EmotionRecognizer:

    def __init__(self, model=None, **kwargs):
        
        # emotions
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy" , "calm", "angry"])
        # make sure that there are only available emotions
        self._verify_emotions()
        # audio config
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)
        # datasets
        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.override_csv = kwargs.get("override_csv", True)
        self.verbose = kwargs.get("verbose", 1)

        self.tess_ravdess_name = kwargs.get("tess_ravdess_name", "tess_ravdess.csv")

        self.verbose = kwargs.get("verbose", 1)

        # set metadata path file names
        self._set_metadata_filenames()
        # write csv's
        self.write_csv()

        # boolean attributes
        self.data_loaded = False
        self.model_trained = False

        # model
        self.model = model

    def _set_metadata_filenames(self):

        train_desc_files, test_desc_files = [], []
        if self.tess_ravdess:
            train_desc_files.append(f"train_{self.tess_ravdess_name}")
            test_desc_files.append(f"test_{self.tess_ravdess_name}")

        # set them to be object attributes
        self.train_desc_files = train_desc_files
        self.test_desc_files  = test_desc_files

    def _verify_emotions(self):
        cnt=0
        for emotion in self.emotions:
            cnt+=1
            print(emotion, " ", cnt)
            assert emotion in AVAILABLE_EMOTIONS, "Emotion not recognized."

    def write_csv(self):

        for train_csv_file, test_csv_file in zip(self.train_desc_files, self.test_desc_files):
           
            if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
                if not self.override_csv:
                    continue
            elif self.tess_ravdess_name in train_csv_file:
                write_tess_ravdess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated TESS & RAVDESS DB CSV File")


    def load_data(self):

        if not self.data_loaded:
            result = load_data(self.train_desc_files, self.test_desc_files, self.audio_config, self.classification,
                                emotions=self.emotions, balance=self.balance)
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.balance = result["balance"]
            if self.verbose:
                print("[+] Data loaded")
            self.data_loaded = True

    