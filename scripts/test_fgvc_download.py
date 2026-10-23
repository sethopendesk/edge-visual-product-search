from datasets import load_dataset
from pathlib import Path

dataset = load_dataset(
    "Donghyun99/FGVC-Aircraft",
    split="train",
    streaming=True
)

example = next(iter(dataset))

# Convert the numeric class ID to an aircraft name
variant_id = example["variation"]
variant_name = dataset.features["variation"].int2str(variant_id)

print("Aircraft:", variant_name)

out_dir = Path("data/fgvc_aircraft")
out_dir.mkdir(parents=True, exist_ok=True)

image = example["image"].convert("RGB")
image.save(out_dir / "test_aircraft.jpg")

print("Saved:", out_dir / "test_aircraft.jpg")