import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self, suppress_tokens: list[int] | None = None):
        super().__init__()
        if suppress_tokens:
            self.register_buffer(
                "_suppress_ids",
                torch.tensor(suppress_tokens, dtype=torch.long),
            )
        else:
            self._suppress_ids = None

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_k: int = 50, top_p: float = 1.0):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        if self._suppress_ids is not None:
            logits[:, self._suppress_ids] = float("-inf")

        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        top_k_logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(top_k_logits, dim=-1)
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        return sample_tokens
