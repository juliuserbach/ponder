"""Dense GenEval reward of Flow-GRPO, scored by its reward server (geneval_server/).

The server detects the requested objects with Mask2Former and returns, per
image, the mean of the object-count, color and position requirements met.
The wire format is Flow-GRPO's: JPEG images and a pickled request, so only
run the server on a trusted network.
"""

import io
import pickle

import requests
import torch
from PIL import Image


class GenEval:
    def __init__(self, url="http://127.0.0.1:18085", batch=8):
        self.url = url
        self.batch = batch

    def __call__(self, images, metadata) -> torch.Tensor:
        """Scores [B] for float images [B,3,H,W] in [0, 1] and their GenEval metadata."""
        arrays = images.float().mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        jpegs = []
        for array in arrays:
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG")
            jpegs.append(buffer.getvalue())
        scores = []
        for i in range(0, len(jpegs), self.batch):
            request = {"images": jpegs[i:i + self.batch], "meta_datas": list(metadata[i:i + self.batch]),
                       "only_strict": True}
            response = requests.post(self.url, data=pickle.dumps(request), timeout=120)
            response.raise_for_status()
            scores += pickle.loads(response.content)["scores"]
        return torch.tensor(scores, dtype=torch.float32, device=images.device)
