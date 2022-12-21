import os

def convert_audio(audio_path, target_path, remove=False):

    v = os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    if remove:
        os.remove(audio_path)
    return v




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Convert ( compress ) wav files to 16MHz and mono audio channel ( 1 channel )
                                                    This utility helps for compressing wav files for training and testing""")
    parser.add_argument("audio_path", help="Folder that contains wav files you want to convert")
    parser.add_argument("target_path", help="Folder to save new wav files")
    parser.add_argument("-r", "--remove", type=bool, help="Whether to remove the old wav file after converting", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("The audio_path file you specified isn't appropriate for this operation")
