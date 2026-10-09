# Table Composition

Like prose_synthesis, but also renders a TableBlock when the gathered facts
are tabular -- the same fact name repeated across several entities (customers,
orders, months). The planner should choose this for "list...", "which
customers...", "show me every...", "rank..." and "break down by..." questions.
Rendering lives in app/skills/output.py; this file is a manifest-loader
placeholder.
