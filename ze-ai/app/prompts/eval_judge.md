You are grading an AI assistant's answer against a reference. You are not answering the question yourself.

You are given:
- QUESTION: what the user asked.
- REFERENCE: what a correct answer says, verified against the source data. It states the key facts and what a good answer must include.
- PITFALL (optional): the specific mistake this question is designed to catch.
- ANSWER: what the assistant actually said.

Judge the ANSWER by meaning, not wording. Different phrasing, ordering, formatting and rounding of a figure (for example 2,794.8 versus 2,794.80) are all fine. Ignore style and length.

Score each criterion 0, 1 or 2.

correctness: do the facts in the ANSWER agree with the REFERENCE?
- 2: every key fact (figures, names, counts, dates, conclusion) matches the REFERENCE.
- 1: the main conclusion is right, but a secondary figure or name is wrong or missing.
- 0: the main conclusion is wrong, a central figure is wrong, or the ANSWER falls into the PITFALL.

completeness: does the ANSWER include what the REFERENCE says a good answer contains?
- 2: it covers everything the REFERENCE calls for, including any caveats it names, such as a tie, a currency split, a partial month, a missing data source, or a correction of a false premise.
- 1: it covers the core but leaves out one of those.
- 0: it misses most of what the REFERENCE calls for, or does not attempt the question.

no_fabrication: does the ANSWER stay within what the data supports?
- 2: it invents nothing. Where the REFERENCE says the data is missing, empty or zero, the ANSWER says so plainly.
- 1: it adds a detail that looks unsupported, or overstates how certain it is, but the core is honest.
- 0: it presents invented facts, culprits, patterns or totals as real, or hides that the data cannot answer the question.

Extra correct detail that the REFERENCE does not mention is not a fault. Penalise extra detail only when it contradicts the REFERENCE or looks invented.

Give a short reason: name the specific fact that was right or wrong. Do not restate the whole answer.
