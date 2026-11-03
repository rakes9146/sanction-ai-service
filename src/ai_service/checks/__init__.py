"""
checks/ - the CODE that verifies the AI.  Every function here is a "pure function": the same input always gives the same output,
and it touches no database, no Kafka, no LLM.  That makes each of them easy to understand and easy to test.

    attribute_compare.py  PA-02   compare DOB / nationality / city / identifiers  -> MATCH | MISMATCH | UNKNOWN
    evidence.py           RC-02   is the AI's quote really in the payment text?
    scoring.py            RC-06/07/09   hit score, band, effective band
    floors.py                     safety floors only ever RAISE the review level
    qa_sampling.py        G-10    which cleared payments are also queued for a human to check
    ownership.py          L2-03   ownership statements: AI lists them, code judges them
"""
