import numpy as np
import torch

from llm.model import TransformerConfig, TransformerLM
from llm.optim import AdamW
from llm.training import Trainer, TrainingConfig, cross_entropy_loss


def test_forward_shape():
    cfg = TransformerConfig(vocab_size=300, max_seq_len=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    model = TransformerLM(cfg)
    x = torch.randint(0, 300, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 300)


def test_cross_entropy_nonnegative():
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    loss = cross_entropy_loss(logits, targets)
    assert loss.item() >= 0


def test_training_and_checkpoint_roundtrip(tmp_path):
    cfg = TransformerConfig(vocab_size=64, max_seq_len=8, d_model=16, n_layers=1, n_heads=4, d_ff=32)
    model = TransformerLM(cfg)
    optim = AdamW(model.parameters(), lr=1e-3)

    arr = np.random.randint(0, 64, size=512, dtype=np.int32)
    tcfg = TrainingConfig(
        batch_size=4,
        seq_len=8,
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    trainer = Trainer(model, optim, arr, arr, tcfg)
    logs = trainer.train()
    assert logs

    ckpt = trainer.save_checkpoint()

    model2 = TransformerLM(cfg)
    optim2 = AdamW(model2.parameters(), lr=1e-3)
    trainer2 = Trainer(model2, optim2, arr, arr, tcfg)
    trainer2.load_checkpoint(ckpt)
    assert trainer2.step == trainer.step
