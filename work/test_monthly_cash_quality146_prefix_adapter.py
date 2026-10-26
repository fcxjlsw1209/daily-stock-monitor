import unittest
from unittest.mock import patch
import numpy as np
import monthly_cash_quality146_prefix_adapter as a


class PrefixTests(unittest.TestCase):
    def test_all123_prefixes_preserve_original_bytes(self):
        values=np.arange(122*49,dtype=float).reshape(122,49)
        for n in range(123):
            self.assertEqual(a.prefix_view(values,n).tobytes(),values[:n].tobytes())
        self.assertEqual(a.prefix_view(values,122).tobytes(),values.tobytes())

    def test_no_aliasing_or_input_modification(self):
        values=np.ones((122,49))
        out=a.prefix_view(values,88);out[:]=9.
        self.assertTrue((values==1.).all())

    def test_actual_backend_adapter_passes_original_prefix_argument(self):
        values=np.arange(122*49,dtype=float).reshape(122,49)
        backend=object.__new__(a.PrefixBackend)
        with patch.object(a.original.Backend,"pre",return_value=values) as mock:
            self.assertEqual(backend.pre({},88).tobytes(),values[:88].tobytes())
            mock.assert_called_once_with({},88)

    def test_invalid_shape_dtype_or_prefix_rejected(self):
        for values,n in ((np.zeros((88,49)),88),(np.zeros((122,48)),88),
                         (np.zeros((122,49),dtype=int),88),
                         (np.zeros((122,49)),-1),(np.zeros((122,49)),123),
                         (np.zeros((122,49)),True)):
            with self.assertRaises(ValueError):a.prefix_view(values,n)


if __name__=="__main__":
    unittest.main()
