import pathlib,hashlib,ast
R=pathlib.Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
p=R/'experiments/robotwin/fastwam_policy/deploy_policy.py'
s=p.read_text();before=s
old='''        if self.replan_steps % self._temporal_subframes != 0:
            raise ValueError(
                "Temporal full-KV requires replan_steps divisible by four, "
                f"got {self.replan_steps}"
            )
'''
new='''        self._temporal_sample_offsets = None
        if self.replan_steps % self._temporal_subframes != 0:
            if os.environ.get("RMBENCH_ALLOW_NONQUARTER_REPLAN") != "1":
                raise ValueError(
                    "Non-quarter replan requires explicit "
                    "RMBENCH_ALLOW_NONQUARTER_REPLAN=1"
                )
            if self.replan_steps < self._temporal_subframes:
                raise ValueError("Replan must allow four distinct observed frames")
            # Keep four causal RGB samples per replan interval. The final
            # sample is the current observation, appended at inference time.
            self._temporal_sample_offsets = tuple(
                (k * self.replan_steps + self._temporal_subframes - 1)
                // self._temporal_subframes
                for k in range(1, self._temporal_subframes)
            )
            logger.info("Replan=%d temporal RGB offsets=%s; current=%d",
                        self.replan_steps, self._temporal_sample_offsets,
                        self.replan_steps)
'''
assert s.count(old)==1;s=s.replace(old,new)
old='''            and self.step_count % self._temporal_sample_stride == 0
'''
new='''            and (
                self.step_count % self.replan_steps in self._temporal_sample_offsets
                if self._temporal_sample_offsets is not None
                else self.step_count % self._temporal_sample_stride == 0
            )
'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s)
# Exercise the exact scheduling rule across 100 replans, including boundaries.
for n in (4,8,10,12,16):
 offsets=None if n%4==0 else tuple((k*n+3)//4 for k in range(1,4))
 frames=[]
 for step in range(n*100+1):
  if step%n==0:
   frames.append(step);assert (len(frames)-1)%4==0
  elif (step%n in offsets if offsets is not None else step%(n//4)==0):frames.append(step)
 assert len(frames)==401 and len(set(frames))==401
 if n==10:assert frames[:9]==[0,3,5,8,10,13,15,18,20]
 if n==16:assert frames==list(range(0,1601,4))
backup=R/'runtime/deploy_policy_before_press_replan10_20260928.py';assert not backup.exists();backup.write_text(before)
p.write_text(s);print('Patched opt-in non-quarter replan; sampling checks passed; baseline16 unchanged')
