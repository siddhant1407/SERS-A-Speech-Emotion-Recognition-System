
from deep_emotion_recognition import DeepEmotionRecognizer

deeprec = DeepEmotionRecognizer(emotions=["sad", "neutral", "happy" , "calm", "angry"], n_rnn_layers=3, n_dense_layers=3, rnn_units=128, dense_units=128)


deeprec.train()
# get the accuracy
print(deeprec.test_score() * 100)
# predict audio sample

prediction = deeprec.predict('data/validation/Actor_08/03-02-03-02-01-02-08_happy.wav')
print(f"Prediction: {prediction}")