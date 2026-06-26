# Literature positioning: state representation and the visual-grounding departure

A draft paragraph for the dissertation, positioning the state mechanism against the
text-only LLM allocation literature, with the key citation. Written to be edited
into the methods or related-work section.

---

## Draft paragraph

The symbolic state representation used here follows the structure established by
recent LLM-based multi-robot task allocation work. SMART-LLM (Kannan et al., 2024)
encodes robot skills and object properties as structured, code-like dictionaries
rather than free-form prose, on the grounds that a structured syntax is easier for a
language model to parse and reduces prompt length; their ablations show that removing
this structure degrades allocation success. The allocator in this work adopts the
same convention, presenting each arm's capabilities (reach, payload, gripper type,
grip force) and each object's symbolic attributes (position, mass, footprint) as a
dictionary-style block, and matches tasks to arms on a capability basis in the manner
those systems describe. The choice of baselines likewise follows this literature:
random and rule-based allocators are the standard comparison points, and SMART-LLM's
own results provide an empirical precedent for the central concern of this study,
namely that a symbolic-only allocator falters precisely when the discriminating
information is an object property it cannot access.

The present work departs from that literature in one deliberate respect, and this
departure is the contribution. In text-only systems such as SMART-LLM, any property
relevant to allocation is, in principle, available to the allocator as a field in the
symbolic state; were fragility relevant, it would simply be encoded as an additional
dictionary entry. This study does the opposite by design. The discriminating
property, object fragility, is withheld from the symbolic state entirely and made
available only through the visual channel, so that a vision-language allocator can
act on it only by grounding what it sees rather than by reading a symbol. The state
extractor enforces this separation programmatically: object identifiers are
neutralised, attributes that could correlate with fragility are held constant across
objects, and a cue-leak check rejects any fragility-related field before it can reach
the allocator's view. The symbolic-only condition therefore functions as a measured
floor, and the difference between it and the vision-grounded condition isolates the
contribution of visual grounding to the allocation decision. In short, the apparatus
follows the SMART-LLM line in its state structure, capability matching, and baselines,
and departs from it precisely by moving the discriminating cue out of the symbolic
state and into perception.

---

## Citation

Kannan, S. S., Venkatesh, V. L. N., and Min, B.-C. (2024). SMART-LLM: Smart
Multi-Agent Robot Task Planning using Large Language Models. In 2024 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS). arXiv:2309.10062.

## Notes for your own use
- Verify the exact citation format against your department's required style, and
  confirm the venue/year line against the published version (the arXiv is 2309.10062).
- The paragraph asserts what SMART-LLM found in its ablation (structure helps); if you
  cite that specific claim, point to their ablation table rather than the abstract.
- This is a draft in your register: no em dashes, plain academic prose. Edit freely so
  the voice is unmistakably yours before it goes in the dissertation.
