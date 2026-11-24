"""HPSv2.1: cosine similarity of the image and prompt embeddings (the official
benchmark reports 100 times its mean). Weights: xswu/HPSv2, HPS_v2.1_compressed.pt."""

import torch
from torch.nn import functional as F

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def clip_pixels(images):
    """CLIP input for float images in [0, 1]: antialiased bicubic resize to 224, normalized."""
    pixels = F.interpolate(images.float(), size=(224, 224), mode="bicubic", align_corners=False,
                           antialias=True)
    return (pixels - pixels.new_tensor(MEAN)[:, None, None]) / pixels.new_tensor(STD)[:, None, None]


class HPSv2:
    def __init__(self, checkpoint, device, batch=64):
        import open_clip

        self.model = open_clip.create_model("ViT-H-14", pretrained=None, precision="fp32",
                                            device=device, output_dict=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.batch = batch

    @torch.no_grad()
    def __call__(self, images, prompts) -> torch.Tensor:
        """Scores [B] for float images [B,3,H,W] in [0, 1] and one prompt per image."""
        return torch.cat([self._score(images[i:i + self.batch], prompts[i:i + self.batch])
                          for i in range(0, len(prompts), self.batch)])

    def _score(self, images, prompts):
        image = F.normalize(self.model.encode_image(clip_pixels(images)).float(), dim=-1)
        unique = list(dict.fromkeys(prompts))
        text = self.model.encode_text(self.tokenizer(unique).to(images.device)).float()
        text = F.normalize(text, dim=-1)[[unique.index(prompt) for prompt in prompts]]
        return (image * text).sum(dim=-1)
