# BMU knowledge base — put the real documents here

Drop the official BMU placement documents into this folder as `.pdf`, `.md` or `.txt`:

- BMU placement policy
- Placement eligibility rules
- Placement guidelines / instructions
- Placement FAQs
- Any other official placement document

Then run:

```
python scripts/ingest.py --reset
```

or press **Ingest folder** in the web UI.

## About the SAMPLE file in this folder

`SAMPLE_placement_policy.md` is **fabricated placeholder text**, written only so the
pipeline can be demonstrated before the real documents arrive. Every rule and number
in it is invented and none of it reflects actual BMU policy.

**Delete it as soon as you add the real documents**, then re-run
`python scripts/ingest.py --reset` so no invented rule can ever be retrieved and
quoted back as if it were policy.

`README.md` files are skipped by the ingester, so this file is never indexed.
