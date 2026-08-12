from CoScientist.integrations.codesynapse.artifacts import (
    INLINE_ARTIFACT_LIMIT_BYTES,
    INLINE_TRACE_PAYLOAD_LIMIT_BYTES,
    payload_delivery_mode,
)


def test_payload_delivery_mode_uses_existing_codesynapse_thresholds():
    assert payload_delivery_mode(INLINE_ARTIFACT_LIMIT_BYTES, artifact=True) == "inline"
    assert payload_delivery_mode(INLINE_ARTIFACT_LIMIT_BYTES + 1, artifact=True) == "reference"
    assert payload_delivery_mode(INLINE_TRACE_PAYLOAD_LIMIT_BYTES + 1, artifact=False) == "reference"
