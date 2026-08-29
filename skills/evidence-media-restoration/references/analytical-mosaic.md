# Analytical mosaic from fragmented sources

## Purpose

Reconstruct the smallest fact pattern supported by heterogeneous fragments: images, video frames, metadata, documents, public posts, diagrams, and technical publications. Coherence is not proof; provenance and independence are.

## Evidence graph

Represent the case as a graph with these node types:

- `artifact`: the exact file/page/post/frame;
- `manifestation`: a crop, encode, screenshot, mirror, or derivative;
- `observation`: a directly visible feature;
- `claim`: a proposition to test;
- `entity`: equipment, person, organization, place, or event;
- `transformation`: extraction, alignment, enhancement, or inference step.

Use edges such as `derived_from`, `visible_in`, `supports`, `contradicts`, `same_recording_as`, `mentions`, and `located_at`. Store source location, retrieval date, hash when possible, and evidence class on every edge.

## Dependence control

Group reposts, copied captions, alternate encodes, and screenshots of one asset into a single witness cluster. Confidence must not increase merely because one source was copied many times.

For claim `C` and evidence clusters `E_i`, a Bayesian update may be written

`P(C|E) proportional to P(C) product_i P(E_i|C)`

only after dependencies are modelled. Do not multiply likelihoods for dependent fragments.

## Alternative hypotheses

For each important claim, write at least two plausible alternatives and an evidence matrix:

| Evidence | H1 | H2 | H3 | Independence cluster |
|---|---:|---:|---:|---|

Prefer evidence that discriminates between hypotheses. Record contradictions and missing expected traces. Do not convert absence of an online trace into proof of absence.

## Minimal reconstruction rule

Promote a claim to `RECONSTRUCTED_MINIMUM` only when:

1. all necessary observations have exact provenance;
2. source dependencies are identified;
3. at least one alternative is materially weakened by the observations;
4. no unsupported detail is required to connect the chain;
5. uncertainty and unresolved alternatives are stated.

## Matrix and graph fusion

Entity-relation matrices or graph completion can rank missing-link hypotheses, but the predicted links are `MODEL_SUGGESTION` until confirmed by an artifact. Matrix tri-factorization and graph neural methods are discovery aids, not evidence authority.

Sheaf-style consistency checks can formalize whether local observations agree on overlaps. Use them only when the overlap maps and local state spaces are explicitly defined; otherwise the terminology adds no evidentiary value.

## Deliverable

Provide:

- artifact inventory and hashes;
- manifestation/witness clusters;
- observation table with exact locations;
- claim-evidence graph or table;
- alternatives and contradictions;
- accepted minimal reconstruction;
- unresolved gaps and the next discriminating artifact to seek.

Relevant foundations: Hall and Llinas on multisensor data fusion; Liggins, Hall, and Llinas, *Handbook of Multisensor Data Fusion*; W3C PROV-O; Heuer, *Psychology of Intelligence Analysis*; Garfinkel on file carving and fragment reconstruction.
