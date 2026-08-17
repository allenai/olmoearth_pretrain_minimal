import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import (
    get_1d_sincos_pos_encoding,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (
    CompositeEncodings,
    Encoder,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality


def test_composite_encodings_compute_temporal_positions_on_the_fly() -> None:
    """CompositeEncodings supports sequences longer than deprecated config value."""
    encodings = CompositeEncodings(
        embedding_size=16,
        supported_modalities=[Modality.SENTINEL2_L2A],
        max_sequence_length=3,
    )
    assert encodings.pos_embed.shape == (3, 4)

    tokens = torch.zeros((1, 2, 2, 5, Modality.SENTINEL2_L2A.num_band_sets, 16))
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020], [3, 2, 2020], [4, 3, 2020], [5, 4, 2020]]]
    )

    output = encodings(
        {Modality.SENTINEL2_L2A.name: tokens},
        timestamps=timestamps,
        patch_size=4,
    )

    encoded = output[Modality.SENTINEL2_L2A.name]
    assert encoded.shape == tokens.shape
    assert torch.isfinite(encoded).all()

    expected_time = get_1d_sincos_pos_encoding(torch.arange(5), 4)
    actual_time = encoded[0, 0, 0, :, 0, 4:8]
    assert torch.allclose(actual_time, expected_time, atol=1e-5)


def test_composite_encodings_loads_temporal_position_table() -> None:
    """Strict state dict loading restores the frozen temporal table."""
    encodings = CompositeEncodings(
        embedding_size=16,
        supported_modalities=[Modality.SENTINEL2_L2A],
        max_sequence_length=3,
    )
    state_dict = encodings.state_dict()
    state_dict["pos_embed"] = torch.zeros((3, 4))

    encodings.load_state_dict(state_dict, strict=True)
    assert torch.equal(encodings.pos_embed, torch.zeros((3, 4)))


def test_band_dropout_disabled_by_default() -> None:
    """Test that Encoder leaves band dropout disabled at construction."""
    encoder = Encoder(
        embedding_size=8,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.1,
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        max_sequence_length=12,
        band_dropout_rate=0.5,
        random_band_dropout=True,
    )
    # Configured rate is stored on the encoder but the patch embeddings
    # start with rate 0.0 so band dropout is inactive until enabled.
    assert encoder.band_dropout_rate == 0.5
    assert encoder.patch_embeddings.band_dropout_rate == 0.0


def test_enable_band_dropout() -> None:
    """Test Encoder.enable_band_dropout activates the configured rate."""
    encoder = Encoder(
        embedding_size=8,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.1,
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        max_sequence_length=12,
        band_dropout_rate=0.5,
        random_band_dropout=True,
    )
    encoder.enable_band_dropout()
    assert encoder.patch_embeddings.band_dropout_rate == 0.5
