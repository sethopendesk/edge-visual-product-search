from datasets import load_dataset

dataset = load_dataset("besartshyti/facepass_eval")

train = dataset["train"]
test = dataset["test"]

print("Training images:", len(train))
print("Testing images:", len(test))

print("Example identity:", train[0]["label"])
print("Image size:", train[0]["image"].size)

train[0]["image"].show()