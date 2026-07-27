"""PI0.5 policy server — uses LeRobot's official loading functions.

Matches the lerobot-eval inference pipeline exactly:
  obs → preprocessor → select_action → postprocessor → action

Usage (on AutoDL server):
    source ~/autodl-tmp/venv_lerobot/bin/activate
    export HF_HOME=/root/autodl-tmp/.hf_cache HF_HUB_OFFLINE=1
    python policy_server.py --checkpoint ~/autodl-tmp/output_my_data/checkpoints/last/pretrained_model

Local client connects via SSH tunnel:
    ssh -p 12956 -L 8765:localhost:8765 root@connect.westd.seetacloud.com
"""

import argparse
import base64
import io
import json
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import numpy as np
import torch
from PIL import Image


def build_obs_batch(
    obs_data: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Decode images + state from client JSON → batch dict. Tokenizer handles task."""
    front = _decode_image(obs_data["front_image"]).to(device)
    wrist = _decode_image(obs_data["wrist_image"]).to(device)
    state = torch.tensor(obs_data["state"], dtype=torch.float32, device=device)

    return {
        "observation.images.front": front,
        "observation.images.wrist": wrist,
        "observation.state": state.unsqueeze(0),
        "task": obs_data.get("task", "move to the red cube and pick it up"),
    }


def _decode_image(b64_str: str) -> torch.Tensor:
    """base64 JPEG → (1, C, H, W) float32 tensor in [0, 1]."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class PolicyHandler(BaseHTTPRequestHandler):
    policy = None
    device = None
    preprocessor = None
    postprocessor = None

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(content_length))

            batch = build_obs_batch(data, self.device)

            return_features = data.get("return_features", False)

            if return_features:
                # Grab vision encoder output via hook
                vision_features = []
                def hook_fn(module, input, output):
                    vision_features.append(output.detach().flatten(1).mean(dim=1).cpu().numpy())

                # Find vision_tower and attach hook
                vision_module = None
                for n, m in self.policy.named_modules():
                    if "vision_tower" in n and "vision_model" in n:
                        vision_module = m
                        print(f"[server] hooked: {n}")
                        break

                if vision_module is None:
                    result = {"error": "vision_tower not found"}
                else:
                    handle = vision_module.register_forward_hook(hook_fn)
                    batch = self.preprocessor(batch)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        self.policy.select_action(batch)
                    handle.remove()
                    if vision_features:
                        result = {"features": [vision_features[0][0].tolist(), vision_features[0][1].tolist()]}
                    else:
                        result = {"error": "no features captured"}
            else:
                batch = self.preprocessor(batch)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    action = self.policy.select_action(batch)
                action = self.postprocessor(action)
                action_np = action.squeeze(0).cpu().numpy()
                result = {"action": action_np.tolist()}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception:
            import traceback
            tb = traceback.format_exc()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(tb.encode())

    def log_message(self, format: str, *args: Any) -> None:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./output_my_data/checkpoints/last/pretrained_model",
        help="Path to the pretrained_model dir with config.json and adapter weights",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import torch
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("cuda")
    checkpoint = args.checkpoint

    # ── 1. Load checkpoint config + base model + LoRA (official pattern) ──
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel

    # Load finetuned config from checkpoint (has front/wrist, 15D state, 8D action)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint

    print(f"Loading LoRA adapter config from: {checkpoint} ...")
    peft_config = PeftConfig.from_pretrained(checkpoint)

    print(f"Loading base model: {peft_config.base_model_name_or_path} with finetuned config ...")
    base_policy = PI05Policy.from_pretrained(
        peft_config.base_model_name_or_path,
        config=cfg,
    )

    print(f"Loading LoRA adapter ...")
    policy = PeftModel.from_pretrained(base_policy, checkpoint, config=peft_config)
    policy = policy.merge_and_unload()

    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Loaded PI0.5+LoRA: {n_params / 1e9:.1f}B params on {device}")
    print(f"Input features: {list(policy.config.input_features.keys())}")
    print(f"Output features: {list(policy.config.output_features.keys())}")

    # ── 2. Preprocessor / postprocessor ──
    print(f"Loading pre/post processors from: {checkpoint} ...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
    )
    print("Preprocessor steps:", [type(s).__name__ for s in preprocessor.steps])

    # ── 3. Inject into handler ──
    PolicyHandler.policy = policy
    PolicyHandler.device = device
    PolicyHandler.preprocessor = preprocessor
    PolicyHandler.postprocessor = postprocessor

    # ── 4. Warmup (no rename — input_features uses front/wrist) ──
    print("Warming up ...")
    dummy = {
        "observation.images.front": torch.randn(1, 3, 480, 640, device=device),
        "observation.images.wrist": torch.randn(1, 3, 480, 640, device=device),
        "observation.state": torch.randn(1, 15, device=device),
        "task": "move to the red cube and pick it up",
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dummy = preprocessor(dummy)
        print("After preprocessor keys:", list(dummy.keys())[:15])
        policy.select_action(dummy)
    print("Warmup OK. Ready on port", args.port)

    server = HTTPServer(("0.0.0.0", args.port), PolicyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
