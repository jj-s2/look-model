from scripts.extract_gmdcsa24_ultralytics_pose import _cache_name, parse_class_segments, pad_or_sample_pose


def test_parse_class_segments_extracts_official_intervals():
    segments = parse_class_segments("Falling (FW)[0.7 to 4]; Standing[0 to 0.6]")
    assert segments[0] == ("fall", 0.7, 4.0)
    assert segments[1] == ("adl", 0.0, 0.6)


def test_pad_or_sample_pose_keeps_masked_shape():
    import numpy as np
    pose = np.ones((3, 17, 3), dtype=np.float32)
    result = pad_or_sample_pose(pose, frames=8)
    assert result.shape == (8, 17, 3)
    assert result[0, 0, 2] == 1.0


def test_cache_name_keeps_same_stem_across_categories_unique():
    assert _cache_name("subject-1", "ADL", "01") != _cache_name("subject-1", "Fall", "01")
