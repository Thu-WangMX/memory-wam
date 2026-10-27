"""Tests for phase alignment, causal lookup, and train-stat normalization."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from extract_labels import phase_arrays
from label_store import DreamDojoLabelStore


class ContractTests(unittest.TestCase):
    def test_demo_boundary(self):
        phase, valid = phase_arrays(6, 3)
        np.testing.assert_array_equal(phase, [1,1,1,2,3,3])
        np.testing.assert_array_equal(valid, [False,True,True,False,True,True])

    def test_execution_only(self):
        phase, valid = phase_arrays(4, 0)
        np.testing.assert_array_equal(phase, [3,3,3,3])
        np.testing.assert_array_equal(valid, [False,True,True,True])

    def test_lookup_never_leaks_future_or_boundary_or_other_episode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            values=np.arange(12*64,dtype=np.float32).reshape(12,64)
            np.save(p/'latents.npy',values)
            np.savez(p/'frame_index.npz',valid=np.tile([False,True,True,False,True,True],2),
                     episode_index=np.repeat([7,9],6),frame_index=np.tile(np.arange(6),2))
            (p/'manifest.json').write_text(json.dumps({'status':'complete','normalization':{
                'all':{'mean':[1]*64,'std':[2]*64}}}))
            store=DreamDojoLabelStore(p)
            labels,mask=store.past_targets(9,[-1,0,1,2,3,4,5,6],observed_through=4)
            np.testing.assert_array_equal(mask,[False,False,True,True,False,True,False,False])
            np.testing.assert_array_equal(labels[~mask],0)
            np.testing.assert_array_equal(labels[mask],(values[[7,8,10]]-1)/2)
            with self.assertRaises(ValueError):
                store.past_targets(9,[1],observed_through=6)


if __name__=='__main__':
    unittest.main()
