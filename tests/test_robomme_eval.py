"""Exercise client batching and native simulator-error termination."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np


class EvalClientTests(unittest.TestCase):
    def test_error_without_observation_preserves_completed_chunk(self):
        calls=[]
        contract={'semantic_tasks':['fake']}

        class Client:
            def __init__(self,**kwargs): pass
            def get_server_metadata(self): return contract
            def reset(self): return {'reset_finished':True}
            def infer(self,inputs):
                calls.append(inputs)
                return {'actions':np.zeros((16,8),np.float32)}

        class Env:
            steps=0
            backend=types.SimpleNamespace(render_device=types.SimpleNamespace(pci_string='fake'))
            @property
            def unwrapped(self): return self
            def reset(self): return {'front_rgb_list':[0]}, {'task_goal':'test'}
            def step(self,action):
                self.steps+=1
                if self.steps==17: return None,None,True,False,{'status':'error'}
                return {'front_rgb_list':[self.steps]},0,False,False,{'status':'running'}
            def close(self): self.closed=True

        env=Env()
        class Builder:
            @staticmethod
            def get_task_list(): return ['fake']
            def __init__(self,**kwargs): self.kwargs=kwargs
            def get_episode_num(self): return 20
            def make_env_for_episode(self,index): return env

        modules={
            'challenge_interface.client':types.SimpleNamespace(PolicyClient=Client),
            'robomme.env_record_wrapper':types.SimpleNamespace(BenchmarkEnvBuilder=Builder)}
        source=Path(__file__).resolve().parents[1]/'experiments/robomme/eval.py'
        spec=importlib.util.spec_from_file_location('robomme_eval_under_test',source)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,modules): spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'contract.json').write_text(json.dumps(contract))
            argv=['eval.py','--task','fake','--contract',str(root/'contract.json'),
                  '--output',str(root/'output'),'--max-steps','1300','--no-save-video']
            with patch.object(sys,'argv',argv),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),contextlib.redirect_stdout(io.StringIO()):
                module.main()
            result=json.loads((root/'output/summary.json').read_text())
        self.assertEqual(result['max_steps'],1300)
        self.assertEqual(result['results'][0]['status'],'error')
        self.assertFalse(result['results'][0]['success'])
        self.assertEqual(result['results'][0]['steps'],17)
        self.assertEqual([x['front_rgb_list'] for x in calls],[[0],list(range(1,17))])
        self.assertEqual([x['is_first_step'] for x in calls],[True,False])
        self.assertTrue(env.closed)


if __name__=='__main__': unittest.main()
