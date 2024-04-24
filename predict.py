# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cog import BasePredictor, Input

MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MODEL_CACHE = ".cache"
MODEL_URL = "https://weights.replicate.delivery/default/d4ve-r/phi-3-mini-128k-instruct"

class WeightsDownloader:
    @staticmethod
    def download_if_not_exists(url, dest):
        if not os.path.exists(dest):
            WeightsDownloader.download(url, dest)

    @staticmethod
    def download(url, dest):
        start = time.time()
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
        print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        WeightsDownloader.download_if_not_exists(MODEL_URL, MODEL_CACHE)

        print("Loading pipeline...")
        torch.set_default_device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_length: int = Input(
            description="Max length", ge=0, le=2048, default=200
        ),
    ) -> str:
        """Run a single prediction on the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_length=max_length)
        result = self.tokenizer.batch_decode(outputs)[0]

        return result
