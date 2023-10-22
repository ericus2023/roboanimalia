from roboflow import Roboflow

rf = Roboflow("wJNP2LlmSsDK4mdrXCiA")
project = rf.workspace().project("bird-v2")
model = project.version(2).model

# infer on a local image
print(model.predict("./bird.jpg", confidence=40, overlap=30).json())

model.predict("./bird.jpg", confidence=40, overlap=30).save("prediction.jpg")