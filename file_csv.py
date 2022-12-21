from emotion_recognition import EmotionRecognizer

emotions = ["sad", "neutral", "happy" , "calm", "angry"]
d = EmotionRecognizer(emotions=emotions)
d.load_data()
exit