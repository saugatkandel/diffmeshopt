"""Test the unified configuration system for regularization weights."""

from diffmeshopt.opt2d.props import (
    AdaptiveRegularizationProps,
    ContourRefinerProps,
    RegularizerDefaults,
    RegularizerType,
)


def test_default_weights():
    """Test that default weights come from RegularizerDefaults."""
    props = ContourRefinerProps()

    # Check static defaults from RegularizerDefaults
    # Note: data loss is not a regularizer (always weight=1.0)
    assert props.get_initial_weight("tangential_laplacian") == 1.0
    assert props.get_initial_weight("normal_consistency") == 1.0
    assert props.get_initial_weight("contour_laplacian") == 0.0
    assert props.get_initial_weight("template_param_anchor") == 0.1
    assert props.get_initial_weight("template_param_laplacian") == 1.0


def test_target_ratios():
    """Test that target ratios are accessible."""
    props = ContourRefinerProps()

    # Check target ratios from RegularizerDefaults
    assert props.get_target_ratio("tangential_laplacian") == 0.1
    assert props.get_target_ratio("normal_consistency") == 0.1
    assert props.get_target_ratio("template_param_anchor") == 0.01
    assert props.get_target_ratio("template_param_laplacian") == 0.05


def test_explicit_overrides():
    """Test that explicit weights override defaults."""
    props = ContourRefinerProps(
        initial_loss_weights={
            "tangential_laplacian": 5.0,  # Override default
            "normal_consistency": 2.0,  # Override default
        }
    )

    assert props.get_initial_weight("tangential_laplacian") == 5.0  # Uses explicit
    assert props.get_initial_weight("normal_consistency") == 2.0  # Uses explicit
    assert props.get_initial_weight("template_param_anchor") == 0.1  # Falls back to default


def test_dict_access():
    """Test that dict-based access works."""
    props = ContourRefinerProps()

    # Read via method
    assert props.get_initial_weight("tangential_laplacian") == 1.0
    assert props.get_initial_weight("normal_consistency") == 1.0
    assert props.get_initial_weight("template_param_anchor") == 0.1

    # Write via dict
    props.initial_loss_weights["tangential_laplacian"] = 3.0
    assert props.get_initial_weight("tangential_laplacian") == 3.0
    assert props.initial_loss_weights["tangential_laplacian"] == 3.0


def test_single_source_of_truth():
    """Test that the single source of truth works correctly."""
    reg_defaults = RegularizerDefaults()
    props = ContourRefinerProps(_reg_defaults=reg_defaults)

    # Verify all enum values are in regularizers
    for loss_type in RegularizerType:
        assert loss_type in reg_defaults.regularizers, f"{loss_type} missing from defaults"

        # Should be able to get weight and ratio
        weight = props.get_initial_weight(loss_type)
        ratio = props.get_target_ratio(loss_type)
        assert isinstance(weight, float)
        assert isinstance(ratio, float)


def test_regularizer_config_structure():
    """Test the RegularizerDefaults structure."""
    defaults = RegularizerDefaults()

    # Check structure
    assert RegularizerType.TANGENTIAL_LAPLACIAN in defaults.regularizers
    assert RegularizerType.NORMAL_CONSISTENCY in defaults.regularizers
    assert RegularizerType.TEMPLATE_PARAM_ANCHOR in defaults.regularizers

    # Check each config has both fields
    tangential_config = defaults.regularizers[RegularizerType.TANGENTIAL_LAPLACIAN]
    assert hasattr(tangential_config, "static_weight")
    assert hasattr(tangential_config, "target_ratio")
    assert tangential_config.static_weight == 1.0
    assert tangential_config.target_ratio == 0.1


def test_enum_usage():
    """Test that RegularizerType enum works for type safety."""
    props = ContourRefinerProps()

    # Test enum usage
    assert props.get_initial_weight(RegularizerType.TANGENTIAL_LAPLACIAN) == 1.0
    assert props.get_initial_weight(RegularizerType.NORMAL_CONSISTENCY) == 1.0
    assert props.get_target_ratio(RegularizerType.TEMPLATE_PARAM_ANCHOR) == 0.01

    # Test string still works (backward compatibility)
    assert props.get_initial_weight("tangential_laplacian") == 1.0
    assert props.get_initial_weight("normal_consistency") == 1.0

    # Test enum value can be used as dict key
    props_with_enum = ContourRefinerProps(
        initial_loss_weights={
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,
        }
    )
    assert props_with_enum.get_initial_weight(RegularizerType.TANGENTIAL_LAPLACIAN) == 5.0
    assert props_with_enum.get_initial_weight("tangential_laplacian") == 5.0  # Both work

    # Test invalid loss name
    assert props.get_initial_weight("invalid_loss") == 0.0

    # Test all enum values are valid
    for loss_type in RegularizerType:
        weight = props.get_initial_weight(loss_type)
        assert isinstance(weight, float)
