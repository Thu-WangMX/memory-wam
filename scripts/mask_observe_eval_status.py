"""Read the persisted campaign snapshot without touching running evaluators."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'evaluate_results/mask_observe_30top3_100_20261005'
results = json.loads((out/'results.json').read_text())
scheduler = json.loads((out/'scheduler.json').read_text())
print('Phase:',results['phase'],'  Preflight:',scheduler['preflight_passed'])
print('Checkpoint    Done/Target  Success  Rate')
for group, data in sorted(results['results'].items(),reverse=True):
    done=data.get('done',0)
    rate=f"{100*data['success']/done:.1f}%" if done else '-'
    print(f"{group}   {done:3}/{data['target']:<3}      {data['success']:3}    {rate}")
print('Selected:',results['selected'])
print('Active:',[(j['group'],j['gpu'],j['pid']) for j in scheduler['active']])
print('Blocked:',scheduler['blocked'])
