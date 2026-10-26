from pathlib import Path
p=Path(__file__).with_name('monthly_formation142_runner.py');s=p.read_text()
s=s.replace("scenarios=arrays['formation_rank'][i,:count] if ready else None", """scenarios=None
            if ready:
                raw=np.asarray(arrays['formation_rank'][i,:count])
                require(all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in raw.flat),'Numeric scenario prefix')
                scenarios=np.asarray(raw,dtype=float)""")
s=s.replace('arm=arm,history_available=ready)',"arm=arm,history_available=arrays['formation_valid'][i] if ready else False)")
s=s.replace("'zero_and_missing_exact_fallback')","'zero_and_missing_exact_fallback',\n    'raw_prices_independently_verified','calendar_overlay_independently_verified',\n    'all_signal_prefixes_verified','formation_block_deletions_independently_verified')")
p.write_text(s)
