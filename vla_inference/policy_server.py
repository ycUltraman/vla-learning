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
import torch.nn as nn
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


# Module-level storage for hook (avoids BaseHTTPRequestHandler class-attr issues)
_HOOK_FEAT = None
_HOOK_COUNT = 0

def _global_feat_hook(m, inp, out):
    global _HOOK_FEAT, _HOOK_COUNT
    _HOOK_COUNT += 1
    _HOOK_FEAT = out.detach()


class PolicyHandler(BaseHTTPRequestHandler):
    policy = None
    device = None
    preprocessor = None
    postprocessor = None
    residual_net = None  # ResidualNet for action correction
    vis_dim = 1024  # action_in_proj output dim

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(content_length))

            batch = build_obs_batch(data, self.device)

            return_features = data.get("return_features", False)
            return_both = data.get("return_both", False)

            if return_both:
                # Hook action_in_proj (matches train_residual.py training features)
                hook_features = []
                def hook_fn(m, inp, out):
                    hook_features.append(out.detach())

                hook_module = None
                for n, m in self.policy.named_modules():
                    if n.endswith("action_in_proj"):
                        hook_module = m
                        break

                if hook_module is None:
                    result = {"error": "action_in_proj not found"}
                else:
                    handle = hook_module.register_forward_hook(hook_fn)
                    batch = self.preprocessor(batch)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        action = self.policy.select_action(batch)
                    handle.remove()
                    action = self.postprocessor(action)
                    action_np = action.squeeze(0).cpu().numpy()
                    feat = hook_features[0].squeeze(0)
                    if feat.dim() == 2: feat = feat.mean(dim=0)
                    result = {
                        "action": action_np.tolist(),
                        "features": feat.cpu().float().numpy().tolist(),
                    }
            elif return_features:
                # Grab vision encoder output via hook
                vision_features = []
                def hook_fn(module, input, output):
                    vision_features.append(output.detach().flatten(1).mean(dim=1).cpu().numpy())

                # Find vision_tower and attach hook
                vision_module = None
                for n, m in self.policy.named_modules():
                    if "vision_tower" in n and "vision_model" in n:
                        vision_module = m
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
                if self.residual_net is not None:
                    global _HOOK_FEAT; _HOOK_FEAT = None
                    batch = self.preprocessor(batch)
                    with torch.no_grad():
                        action = self.policy.select_action(batch)
                    action = self.postprocessor(action)
                    action_np = action.squeeze(0).cpu().numpy()
                    result = {"action": action_np.tolist()}

                    feat = _HOOK_FEAT
                    if feat is not None:
                        feat = feat.squeeze(0)
                        if feat.dim() == 2: feat = feat.mean(dim=0)
                        feat_np = feat.cpu().float().numpy()
                    else:
                        feat_np = np.zeros(self.vis_dim, dtype=np.float32)
                    with torch.no_grad():
                        f = torch.from_numpy(feat_np).float().to(self.device).unsqueeze(0)
                        s = torch.zeros(1, 7, device=self.device)
                        b = torch.from_numpy(action_np[:3]).float().to(self.device).unsqueeze(0)
                        delta = self.residual_net(f, s, b).squeeze(0).cpu().numpy()
                    corrected = action_np.copy()
                    corrected[0] += delta[0]
                    corrected[1] += delta[1]
                    corrected[2] += delta[2]
                    result["corrected_action"] = corrected.tolist()
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


class ResidualNet(nn.Module):
    """Minimal residual network (mirrors train_residual.ResidualNet)."""
    def __init__(self, vis_dim):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(74, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, vis_feat, state7, bc_xyz):
        v = self.vis_proj(vis_feat)
        x = torch.cat([v, state7, bc_xyz], dim=-1)
        raw = torch.tanh(self.net(x))
        scale = torch.tensor([0.02, 0.02, 0.005], device=x.device)
        return raw * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./output_my_data/checkpoints/last/pretrained_model",
        help="Path to the pretrained_model dir with config.json and adapter weights",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--residual-model", default=None,
                        help="Path to residual_net.pt for action correction")
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

    policy = policy.to(device=device, dtype=torch.float32)
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

    # ── 3. Load residual model (optional) ──
    if args.residual_model:
        print(f"Loading residual model from: {args.residual_model} ...")
        # vis_dim from action_in_proj (1024 for PI0.5 base)
        vis_dim = 1024
        residual = ResidualNet(vis_dim).to(device=device, dtype=torch.float32)
        residual.load_state_dict(torch.load(args.residual_model, map_location=device))
        residual.eval()
        PolicyHandler.residual_net = residual
        print("Residual model loaded.")
    else:
        PolicyHandler.residual_net = None

    # ── 4. Register persistent hook on action_in_proj ──
    PolicyHandler.vis_dim = 1024
    for n, m in policy.named_modules():
        if n.endswith("action_in_proj"):
            m.register_forward_hook(_global_feat_hook)
            if hasattr(m, 'out_features'):
                PolicyHandler.vis_dim = m.out_features
            print(f"  Registered persistent hook on: {n}, vis_dim={PolicyHandler.vis_dim}")
            break

    # ── 5. Inject into handler ──
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
    with torch.no_grad(), warnings.catch_warnings():
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
