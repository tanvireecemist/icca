from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from rf_research.model.layers import gradient_reverse
from rf_research.model.losses import supervised_contrastive_loss
from rf_research.model.network import DualDomainRFEncoder


class RFFingerprintModule(L.LightningModule):
    def __init__(
        self,
        config: dict[str, Any],
        num_classes: dict[str, int],
        source_map: dict[str, int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "model": config["model"],
                "trainer": config["trainer"],
                "num_classes": num_classes,
                "source_map": source_map,
            }
        )
        model = config["model"]
        self.encoder = DualDomainRFEncoder(
            stem_width=int(model["stem_width"]),
            model_dim=int(model["model_dim"]),
            transformer_layers=int(model["transformer_layers"]),
            attention_heads=int(model["attention_heads"]),
            embedding_dim=int(model["embedding_dim"]),
            dropout=float(model["dropout"]),
        )
        self.heads = nn.ModuleDict(
            {
                source: nn.Linear(int(model["embedding_dim"]), count)
                for source, count in num_classes.items()
            }
        )
        self.source_map = source_map
        self.index_to_source = {index: source for source, index in source_map.items()}
        self.domain_head = nn.Sequential(
            nn.Linear(int(model["embedding_dim"]), int(model["embedding_dim"]) // 2),
            nn.SiLU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(int(model["embedding_dim"]) // 2, len(source_map)),
        )
        self.supcon_weight = float(model["supcon_weight"])
        self.domain_weight = float(model["domain_weight"])
        self.temperature = float(model["supcon_temperature"])
        self.learning_rate = float(config["trainer"]["learning_rate"])
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.warmup_fraction = float(config["trainer"]["warmup_fraction"])
        self.test_predictions: list[dict[str, float | int]] = []

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.encoder(iq)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        embedding = self(batch["iq"])
        source = batch["source"]
        labels = batch["label"]
        classification_loss = embedding.new_zeros(())
        correct = embedding.new_zeros(())
        total = embedding.new_zeros(())
        predictions = torch.empty_like(labels)
        confidences = torch.empty_like(labels, dtype=embedding.dtype)

        for source_index, source_name in self.index_to_source.items():
            mask = source == source_index
            count = int(mask.sum())
            if not count:
                continue
            logits = self.heads[source_name](embedding[mask])
            source_loss = F.cross_entropy(logits, labels[mask])
            classification_loss = classification_loss + source_loss * (
                count / labels.shape[0]
            )
            probability = logits.softmax(dim=1)
            confidence, predicted = probability.max(dim=1)
            predictions[mask] = predicted
            confidences[mask] = confidence
            source_correct = (predicted == labels[mask]).sum()
            correct = correct + source_correct
            total = total + count
            self.log(
                f"{stage}/{source_name}_accuracy",
                source_correct.float() / count,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=count,
            )

        contrastive_labels = source * 100_000 + labels
        contrastive_loss = supervised_contrastive_loss(
            embedding, contrastive_labels, self.temperature
        )
        source_logits = self.domain_head(gradient_reverse(embedding))
        domain_loss = F.cross_entropy(source_logits, source)
        loss = (
            classification_loss
            + self.supcon_weight * contrastive_loss
            + self.domain_weight * domain_loss
        )
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.shape[0],
        )
        self.log(
            f"{stage}/accuracy",
            correct / total.clamp_min(1),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.shape[0],
        )
        if stage == "train":
            self.log(
                "train/classification_loss",
                classification_loss,
                on_step=False,
                on_epoch=True,
                batch_size=labels.shape[0],
            )
            self.log(
                "train/contrastive_loss",
                contrastive_loss,
                on_step=False,
                on_epoch=True,
                batch_size=labels.shape[0],
            )
            self.log(
                "train/domain_loss",
                domain_loss,
                on_step=False,
                on_epoch=True,
                batch_size=labels.shape[0],
            )
        if stage == "test":
            for index in range(labels.shape[0]):
                self.test_predictions.append(
                    {
                        "record_index": int(batch["record_index"][index].detach().cpu()),
                        "source_index": int(source[index].detach().cpu()),
                        "label": int(labels[index].detach().cpu()),
                        "prediction": int(predictions[index].detach().cpu()),
                        "confidence": float(confidences[index].detach().cpu()),
                    }
                )
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_test_epoch_start(self) -> None:
        self.test_predictions.clear()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=torch.cuda.is_available(),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=max(0.01, self.warmup_fraction),
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

