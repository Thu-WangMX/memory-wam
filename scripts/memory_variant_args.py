"""Emit four shell-safe lines for the canonical Helios variant profile."""
import json,sys
from fastwam.memory.helios_history import memory_variant_profile
p=memory_variant_profile(sys.argv[1])
for key in ('history_sizes','patch_long','patch_mid','anchor_frames'):
    print(json.dumps(p[key],separators=(',',':')))
