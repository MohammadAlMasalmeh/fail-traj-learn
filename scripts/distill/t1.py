import sys
sys.path.insert(0, 'scripts/distill')
import data
rs = data.load_records()
tr, ev = data.make_split(rs)
s = data.load_split()
print(len(tr), len(ev), s['n_failure_total'], s['n_failure_eval'])
m = data.build_student_messages(rs[0])
print(len(m[0]['content']))
print(m[0]['content'][0]['text'][-700:])
print(m[0]['content'][1:6])
