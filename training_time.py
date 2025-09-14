import math
from dataclasses import dataclass

@dataclass
class TrainTime:
    params: int                 # total trainable params (absolute count)
    gpu_memory_gb: float        # per-GPU VRAM
    precision: str = "bf16"     # 'fp16'|'bf16'|'fp32'
    optimizer: str = "adam"     # 'adam'|'adam8bit'|'lion'
    zero_stage: int = 2         # 0|1|2|3 (rough sharding)
    activation_checkpoint: bool = True
    overhead_frac: float = 0.10 # allocator/workspace/comms slack
    # Efficiency knobs
    base_eff: float = 0.35      # cluster utilization before memory penalty

    # Memory and Headroom
    @staticmethod
    def _bytes_per_param_states(precision: str, optimizer: str) -> int:
        p = precision.lower()
        o = optimizer.lower()
        if p in ("fp16", "bf16"):
            if o == "adam8bit": return 10  # fp16 + fp32 master + 8-bit m,v
            return 16                       #  fp16 + fp32 master + m,v fp32
        return 24                          # crude fp32 training fallback

    def _zero_factor(self) -> float:
        return {0: 1.00, 1: 0.55, 2: 0.40, 3: 0.20}.get(int(self.zero_stage), 1.00)

    def _activation_frac(self) -> float:
        return 0.2 if self.activation_checkpoint else 0.5

    def _used_vram_gb(self) -> float:
        bpp = self._bytes_per_param_states(self.precision, self.optimizer)
        zf  = self._zero_factor()
        model_state_bytes = self.params * bpp * zf
        activation_bytes  = model_state_bytes * self._activation_frac()
        total_bytes = (model_state_bytes + activation_bytes) * (1.0 + self.overhead_frac)
        return total_bytes / (1024**3)

    def headroom(self) -> float:
        """Estimate VRAM headroom fraction in [<=0, 1]."""
        used = self._used_vram_gb()
        return (self.gpu_memory_gb - used) / self.gpu_memory_gb

    # Efficiency penalty from memory
    @staticmethod
    def _penalty_piecewise(headroom: float) -> float:
        h = max(-1.0, min(1.0, headroom))
        if h <= 0.0: return 0.0   # OOM
        if h >= 0.25: return 1.00
        if h >= 0.15: return 0.92
        if h >= 0.08: return 0.85
        return 0.75

    @staticmethod
    def _penalty_logistic(headroom: float, k: float = 20.0, midpoint: float = 0.15) -> float:
        h = max(0.0, min(1.0, headroom))
        return 0.75 + 0.25 / (1.0 + math.exp(-k * (h - midpoint)))

    def effective_efficiency(self, smooth: bool = False) -> float:
        h = self.headroom()
        penalty = self._penalty_logistic(h) if smooth else self._penalty_piecewise(h)
        return max(0.0, min(1.0, self.base_eff * penalty))

    # Training Time
    def estimate_time(self, total_flops: float, per_gpu_flops: float, num_gpus: int,
                      smooth: bool = False):
        eff = self.effective_efficiency(smooth=smooth)
        cluster_rate = per_gpu_flops * num_gpus * eff  # FLOPs/s
        seconds = total_flops / cluster_rate if cluster_rate > 0 else float("inf")
        return {
            "used_vram_gb": round(self._used_vram_gb(), 2),
            "headroom": round(self.headroom(), 3),
            "efficiency_used": round(eff, 3),
            "cluster_flops_per_s": cluster_rate,
            "seconds": seconds,
            "hours": seconds / 3600.0,
            "days": seconds / 86400.0,
        }

