"""Metadata-only real binding regression; does not load price panels or NPZ."""
import json
import monthly_calendar_clock248_host as h
h.require(not h.OUT.exists(),"Must precede registration")
h.source_review_check(h.read(h.SOURCE_REVIEW))
print(json.dumps(dict(passed=True,metadata_only=True,prices_loaded=False,npz_decoded=False,source_review_sha256=h.sha(h.SOURCE_REVIEW),context_sha256=h.sha(h.CONTEXT),host_sha256=h.sha(h.ROOT/'work/monthly_calendar_clock248_host.py'))))
