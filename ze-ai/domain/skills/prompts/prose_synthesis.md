# Prose Synthesis

The default output skill: renders findings as cited prose. Not a tool-calling
skill -- see app/skills/output.py for the actual rendering, which builds every
metric block deterministically from validated Facts and only uses a model to
weave multiple skills' narratives together (verifying afterwards that it
invented no citations). This file exists so the manifest loader has a prompt
to validate.
