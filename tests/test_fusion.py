import numpy as np

from hdea.fusion import geometric_pool


def test_geometric_pool_matches_mean_log_probability_readout():
    values = np.log(
        np.asarray(
            [
                [0.60, 0.30, 0.10],
                [0.30, 0.60, 0.10],
                [0.55, 0.35, 0.10],
            ]
        )
    )
    log_probs, probabilities, prediction = geometric_pool(values)
    expected = np.exp(values.mean(axis=0))
    expected /= expected.sum()
    assert np.allclose(probabilities, expected)
    assert np.allclose(np.exp(log_probs), expected)
    assert prediction == int(np.argmax(expected))

