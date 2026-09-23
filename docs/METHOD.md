# Frozen HDEA method

HDEA is a training-free evidence acquisition method for long-video
multiple-choice question answering. The released implementation contains the
two modules evaluated in the paper.

## Module I: Hypothesis-Discriminative Core Acquisition

For candidate-conditioned retrieval scores \(z_i(c_a)\), every local
two-frame packet defines one facility vector

\[
d_i(a,b)=|z_i(c_a)-z_i(c_b)|,\quad a<b.
\]

For a selected packet set \(S\), HDEA maximizes

\[
H(S)=\sum_{a<b}\max_{i\in S}d_i(a,b)
\]

by deterministic greedy marginal gain inside an adaptive temporal partition.
The feasible region, required timestamp contraction, 16-packet budget, and
seven-decimal tie canonicalization are fixed. Selected physical frames are
restored to chronological order before answer scoring.

## Module II: Core-Preserving Nested Refinement

Let \(\pi\) be the frozen answerer's posterior on the 32-frame core. Remaining
candidate frames are split into eight chronological partitions; each action
contains four endpoint-inclusive, evenly spaced frames. The action value is

\[
v_\pi(A)=\sum_{a<b}\pi_a\pi_b\,\frac{1}{|A|}
\sum_{f\in A}|z_f(c_a)-z_f(c_b)|.
\]

The lowest-valued action (earliest on a tie) is retained as the matched
control. The remaining actions are ranked once by descending value, with the
earlier action breaking ties. E40 adds the first two actions and E48 adds the
first four, so E32 is a subset of E40 and E40 is a subset of E48. When fewer
than 32 residual candidate frames remain, both expanded views stop at E32.

The final distribution is the equal-weight geometric pool of the three answer
distributions:

\[
\bar p(c)\propto\exp\left[\frac{1}{3}
\sum_{E\in\{E_{32},E_{40},E_{48}\}}\log p(c\mid E)\right].
\]

No gold answer, correctness signal, subtitle, ASR, captioner, external
generative model, or target-dataset update enters evidence selection.

