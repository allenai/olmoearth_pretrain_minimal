from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import Encoder
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality


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
