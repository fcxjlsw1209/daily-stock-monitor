from pathlib import Path
p=Path(__file__).with_name('test_monthly_formation142_runner.py');s=p.read_text();anchor='    def test_invalid_prefix(self):\n'
new='''    def test_candidate_with_invalid_original_core_falls_back(self):
        self.a['formation_valid'][0,48]=False
        c,s,d=r.choices_all(self.a,self.m,1,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(d[name][0]['reason'],'candidate_history_unavailable')
            self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_nonnumeric_or_boolean_scenario_prefix_fails(self):
        for value in ('0.1',True):
            a=dict(self.a,formation_rank=self.a['formation_rank'].astype(object))
            a['formation_rank'][0,0,0]=value
            with self.assertRaises(ValueError):r.choices_all(a,self.m,1,self.b,POLICY)
'''
assert anchor in s;s=s.replace(anchor,new+anchor);p.write_text(s)
