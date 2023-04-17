import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["right-up left-down down-left right-up"]).to(device)
t = text.unsqueeze(2).unsqueeze(3)
text = t.expand(t.size(0),
                                             t.size(1),
                                             200,
                                             200)
with torch.no_grad():
   # image_features = model.encode_image(image)
    text_features = model.encode_text(text)

   # logits_per_image, logits_per_text = model(image, text)
   # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", 0)  # prints: [[0.9927937  0.00421068 0.00299572]]