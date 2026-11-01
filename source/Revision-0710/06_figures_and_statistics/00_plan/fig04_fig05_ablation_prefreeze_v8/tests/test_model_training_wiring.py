from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

import v8_contract as contract
import v8_data
import v8_inputs
import v8_loss
import v8_model
import v8_training


def test_all_twelve_variants_build_and_optimizer_sets_are_exact() -> None:
    manifests = []
    for variant_id in contract.VARIANT_IDS:
        model = v8_model.build_tiny_production_model(variant_id)
        manifest = v8_training.build_optimizer_manifest(model, variant_id)
        optimizer = v8_training.build_adamw(model, manifest, lr=2e-4, weight_decay=1e-4)
        v8_training.validate_optimizer_exact(model, optimizer, manifest)
        assert manifest["variant_id"] == variant_id
        assert manifest["model_parameter_count"] > 0
        assert manifest["optimizer_parameter_count"] > 0
        assert len(manifest["initial_trainable_sha256"]) == 64
        assert set(manifest["optimizer_parameter_names"]) == set(manifest["trainable_parameter_names"])
        assert not set(manifest["optimizer_parameter_names"]).intersection(
            manifest["frozen_or_unused_parameter_names"]
        )
        manifests.append(manifest)
    assert len(manifests) == 12


def test_variant_specific_optimizer_exclusions() -> None:
    query = v8_training.build_optimizer_manifest(
        v8_model.build_tiny_production_model("fig04_02_no_organ_queries"),
        "fig04_02_no_organ_queries",
    )
    assert any(name.endswith("organ_queries") for name in query["frozen_or_unused_parameter_names"])
    assert not any(name.endswith("organ_queries") for name in query["optimizer_parameter_names"])

    no_prog = v8_training.build_optimizer_manifest(
        v8_model.build_tiny_production_model("fig04_04_no_survival_supervision"),
        "fig04_04_no_survival_supervision",
    )
    assert any(name.endswith("baseline_log_rate") for name in no_prog["frozen_or_unused_parameter_names"])

    no_text = v8_training.build_optimizer_manifest(
        v8_model.build_tiny_production_model("fig05_04_no_event_text"),
        "fig05_04_no_event_text",
    )
    assert any("text_projection" in name for name in no_text["frozen_or_unused_parameter_names"])


def test_independent_head_loss_is_the_production_route() -> None:
    variant = "fig04_07_seven_independent_organ_age_heads"
    model = v8_model.build_tiny_production_model(variant)
    model.eval()
    batch = v8_training.synthetic_batch(batch_size=8, seed=11)
    output = model(**batch.inputs)
    losses, wiring = v8_loss.compute_variant_loss(
        variant,
        output,
        labels=batch.labels,
        interval_edges_months=torch.tensor([0, 12, 24, 36, 48, 60], dtype=torch.float32),
        survival_multiplier=1.0,
        augmented_output=None,
    )
    assert wiring["loss_route"] == "independent_heads_joint_loss"
    assert "independent_age_huber" in losses
    assert "base_huber" not in losses
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.model.independent_age_heads.parameters())


def test_standard_loss_overrides_affect_production_wiring() -> None:
    expected = {
        "fig04_04_no_survival_supervision": (0.0, 0.25, 0.75, 0.1, 0.1),
        "fig04_05_no_strict_offset_shrinkage": (0.5, 0.25, 0.0, 0.1, 0.1),
        "fig04_06_no_consistency_objective": (0.5, 0.25, 0.75, 0.0, 0.1),
    }
    for variant, values in expected.items():
        policy = v8_loss.variant_loss_policy(variant)
        observed = (
            policy["survival_weight"],
            policy["evaluator_gap_cox_weight"],
            policy["strict_offset_weight"],
            policy["consistency_weight"],
            policy["event_dropout_probability"],
        )
        assert observed == values


def test_input_mask_variants_use_production_helper_and_fail_on_holes() -> None:
    batch = v8_training.synthetic_batch(batch_size=8, seed=12)
    for variant in contract.FIG05_VARIANTS:
        transformed = v8_inputs.apply_variant_inputs(batch.inputs, variant)
        v8_inputs.assert_left_aligned(transformed["event_valid_mask"])
    broken = copy.deepcopy(batch.inputs)
    broken["event_valid_mask"][0, 1] = False
    broken["event_valid_mask"][0, 2] = True
    with pytest.raises(ValueError, match="left-aligned"):
        v8_inputs.apply_variant_inputs(broken, "fig05_02_no_hospital_icd")


def test_scheduler_is_warmup_then_cosine_with_fixed_horizon() -> None:
    values = [
        v8_training.learning_rate_multiplier(epoch, warmup_epochs=3, horizon_epochs=60)
        for epoch in (1, 2, 3, 4, 55, 60)
    ]
    assert values[0] < values[1] < values[2]
    assert values[3] == pytest.approx(values[2])
    assert values[-1] > 0.0
    assert values[-2] > 0.0


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_prediction_frame_moves_only_bounded_batches_to_model_device(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.batch_sizes: list[int] = []

        def forward(self, **inputs):
            batch_size = int(inputs["sex"].shape[0])
            self.batch_sizes.append(batch_size)
            assert all(value.device == self.anchor.device for value in inputs.values())
            values = torch.zeros((batch_size, 7), device=self.anchor.device)
            return SimpleNamespace(predicted_age=values + 50.0, risk_score=values)

    participants = tuple(f"p{index}" for index in range(5))
    bundle = v8_data.RoleBundle(
        role="calibration",
        outer_fold=0,
        participant_ids=participants,
        inputs={"sex": torch.tensor([0, 1, 0, 1, 0])},
        labels={
            "chronological_age": torch.arange(50, 55, dtype=torch.float32),
            "strict_healthy_reference": torch.ones(5, dtype=torch.bool),
            "duration_months": torch.ones((5, 7), dtype=torch.float32),
            "event": torch.zeros((5, 7), dtype=torch.bool),
            "at_risk": torch.ones((5, 7), dtype=torch.bool),
        },
    )
    model = CountingModel()
    frame = v8_data.prediction_frame(
        model,
        bundle,
        variant_id=contract.VARIANT_IDS[0],
        run_id=contract.expected_run_ids()[0],
        evaluation_epoch=55,
        device=torch.device(device_name),
        inference_batch_size=2,
    )
    assert model.batch_sizes == [2, 2, 1]
    assert model.anchor.device.type == device_name
    assert len(frame) == 5 * 7
