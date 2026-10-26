"""Pure shape adaptation for the frozen round177 independent delta audit.

The caller pins and supplies the original verification function. This module
does not load data, import an engine, alter a strategy input, or select a shape
based on outcomes. A separate shallow mapping is used only for that verifier.
"""
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _snapshot(original):
    require(isinstance(original, dict), 'Original arrays must be a dictionary')
    require(all(isinstance(v, np.ndarray) for v in original.values()),
            'Original values must remain NumPy arrays')
    return {k: (id(v), v.dtype.str, v.shape, v.strides, v.tobytes(order='C'))
            for k, v in original.items()}


def adapted_verify_deltas(values, deltas, original, frozen_verify, dose=.5):
    """Validate all state weights, copy their representation, invoke old audit.

    Legal forms are (n,), (n,1), and (n,width). Every weight must equal its
    original calm/noncalm value exactly. No averaging, tolerance or fill is used.
    All other entries preserve object identity; original bytes are checked both
    on successful verification and when the supplied audit raises.
    """
    require(callable(frozen_verify) and frozen_verify is not adapted_verify_deltas,
            'A separate frozen verifier callback is required')
    require(type(dose) in (int, float) and dose == .5, 'Frozen half-core dose required')
    require(isinstance(original, dict) and
            {'core', 'core_weight', 'state_calm10'} <= set(original), 'Original array keys required')
    before = _snapshot(original)
    core, weights, calm = (original[k] for k in ('core', 'core_weight', 'state_calm10'))
    require(core.ndim == 2 and core.shape[1] > 0, 'Original core must have shape (n,width)')
    n, width = core.shape
    require(calm.shape == (n,) and calm.dtype == np.dtype(bool), 'Original calm flags must be boolean (n,)')
    require(weights.dtype.kind in 'iuf' and np.isfinite(weights).all(),
            'Original weights must be finite real numeric values')
    require(weights.shape in ((n,), (n, 1), (n, width)), 'Unsupported original weight shape')
    expected = np.where(calm, .50, .52)
    broadcast = expected if weights.ndim == 1 else np.broadcast_to(expected[:, None], weights.shape)
    require(np.array_equal(weights, broadcast), 'Every original weight must exactly match its state')
    normalized = (weights if weights.ndim == 1 else weights[:, 0]).copy()
    require(normalized.shape == (n,) and not np.shares_memory(normalized, weights),
            'Normalized weights must be a separate one-dimensional copy')
    view = dict(original)
    view['core_weight'] = normalized

    def verify_unchanged():
        require(_snapshot(original) == before, 'Original array identity, representation or bytes changed')
        require(list(view) == list(original), 'Verifier array keys changed')
        require(all(view[k] is original[k] for k in original if k != 'core_weight'),
                'Only verifier weight representation may change')
        require(view['core_weight'] is normalized and np.array_equal(normalized, expected),
                'Verifier changed normalized weights')

    try:
        receipt = frozen_verify(values, deltas, view, dose=dose)
    finally:
        verify_unchanged()
    require(isinstance(receipt, dict) and receipt.get('passed') is True, 'Frozen delta audit did not pass')
    require('weight_shape_adapter' not in receipt, 'Shape evidence already present')
    result = dict(receipt)
    result['weight_shape_adapter'] = dict(
        passed=True, original_shape=list(weights.shape), normalized_shape=[n], rows=n, width=width,
        original_dtype=weights.dtype.str, normalized_dtype=normalized.dtype.str,
        validated_weight_cells=int(weights.size), exact_state_weights=True,
        original_dict_unchanged=True, original_arrays_unchanged=True, other_arrays_same_objects=True,
        only_weight_replaced=True, weight_copy_not_shared=True, applied_to_verifier_only=True)
    return result
