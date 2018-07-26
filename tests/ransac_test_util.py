import numpy as np


def confidence(inlier_ratio, n_sample, n_trial):
    bad_sample_prob = 1 - inlier_ratio ^ n_sample
    no_good_sample_prob = bad_sample_prob ^ n_trial
    return 1 - no_good_sample_prob


def trial_count(inlier_ratio, n_sample, confidence=0.9999999980268246):
    """default value of confidence is 6 sigma"""
    no_good_sample_prob = 1 - inlier_ratio ** n_sample

    assert 0.5 <= confidence < 1
    assert 0 <= no_good_sample_prob < 1

    val = np.ceil(np.log(1 - confidence) / np.log(no_good_sample_prob))
    if val == np.inf or val == -np.inf:
        return np.inf
    return int(val)
