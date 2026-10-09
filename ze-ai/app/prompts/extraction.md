You just finished an investigation using tools. Below is the full transcript: the task, every tool call you made (with its REAL tool_call_id), every tool result, and your final narrative.

Produce a SkillResult.

- narrative: your findings in plain language, citing each value as [f_01], [f_02], etc.
- facts: one Fact per concrete claim the narrative depends on -- numbers AND key strings (statuses, plan names, dates, ids). Every Fact's provenance.tool_call_id MUST be one of the real tool_call_id values in the transcript; never invent one. For a value you computed from other facts, set `derivation` and reuse an input fact's tool_call_id for provenance.
- status: "ok" if you fully answered, "partial" if only partly, "error" if tools failed and you have nothing reliable, "denied" if a tool refused you.
- caveats: anything that must travel with the answer (currencies not blended, window is partial, data possibly stale, etc).

CRITICAL -- value_str and value_kind must be right:
- value_str holds ONLY the bare value. A money amount's currency goes in the separate `currency` field, never inside value_str.
    RIGHT: value_str="792.00", currency="USD"
    WRONG: value_str="792.00 USD", currency="USD"
    WRONG: value_str="$792"
- value_kind must match what value_str actually is:
    "decimal" -> ONLY a numeric amount/measure, e.g. "792.00", "4.67"
    "int"     -> ONLY a whole-number count, e.g. "22"
    "date"    -> an ISO date, e.g. "2026-10-10"
    "str"     -> EVERYTHING else: currency codes ("EUR"), states ("SHIPPED"), names ("Acme Ltd"), ids, free text
    "bool"    -> "true"/"false"
  A currency code, a status and a name are ALWAYS "str" -- never "decimal", even when they sit beside a number in your narrative.

NEVER create a fact whose whole value is a currency code or a unit. The currency belongs on the `currency` field of the amount it describes, and the unit on `unit`. A standalone "USD" fact renders as the bare word "USD" wherever it is cited, which produces broken sentences like "charged through USD" and doubled text like "990.00 USD USD".
    RIGHT: one fact -- value_str="990.00", currency="USD", name="order_total"
    WRONG: two facts -- one value_str="990.00" plus another value_str="USD", name="currency"

Cite each fact where it actually belongs in the sentence. A marker placed next to the wrong words reads as nonsense even when the underlying value is correct.

Do not put any number in the narrative that is not backed by a Fact.
