+++
title = "The Teacher Grading Its Own Exam: Why AI Improving Itself Could be Misaligned"
date = 2026-09-03
+++

*context:* There's a comforting story going around about how we'll keep advanced AI aligned as it gets smarter than us: let AI help supervise AI. One version is weak-to-strong generalization, [introduced by OpenAI in 2023](https://arxiv.org/abs/2312.09390): take a weak supervisor (think GPT-2, or honestly, a human) and use it to train a much stronger model. Surprisingly, the strong model often generalizes past the weak supervisor's mistakes, behaving closer to what the supervisor meant than to what its labels actually said. Another is [Constitutional AI](https://arxiv.org/abs/2212.08073), introduced by Anthropic in 2022, which replaces most human feedback with AI feedback: a model critiques and revises responses against a written list of principles, and those judgments become the training signal. Together they're the leading blueprint for how we might keep aligning systems that are already smarter than us in specific domains.

The problem is that this story quietly stretches to cover a very different setup: recursive self-improvement, where a model shapes the values of its own next version. On the whiteboard it's the same diagram (a judge's signal in, a new model out). Constitutional AI already sits halfway there, since the model giving the feedback is usually a close relative of the model being trained. And this isn't an argument about capability. Assume the model never gets any smarter from one round to the next. The worry is only about whose values end up in the training signal, and that depends on who is doing the judging.

*analogy:* Think of it as a classroom. In weak-to-strong, a teacher grades a student who's smarter than they are. The teacher will miss things, but they don't care what grade the student gets. Now picture a teacher grading the exam that they themselves will sit next year, using a rubric they get to reinterpret along the way. Same classroom, same red pen, completely different incentives, and the difference matters a lot more than the diagrams suggest.

## The part nobody draws on the whiteboard

In classic weak-to-strong setups, the teacher and the student are different parties. A human labeler, or a small frozen model, has no stake in what values the strong model ends up with. It judges each answer (honest or not, harmful or not, within the rules or not), and its judgment doesn't change depending on what the student turns into. It can be wrong, but it's wrong in a fixed way, and fixed errors are exactly what the 2023 result showed a strong student can generalize past.

Recursive self-improvement collapses that separation. When a model generates its own alignment signal, critiquing its own answers against a constitution, picking which of its outputs is more harmless, deciding what "within the rules" means in edge cases, the judge is the same lineage as the model being judged. The teacher this round becomes (a version of) the student next round. Two things follow, and neither needs the model to become any more capable:

1. **Errors stop washing out and start compounding.** An external supervisor's blind spots stay where they are. A self-supervisor's blind spots get trained into the next generation, which then judges with the same blind spots, a little more confidently. With no fixed reference point outside the loop, small misreadings of the rules pile up like a photocopy of a photocopy.
2. **The judge has a stake in the verdict.** If the model has any preferences about its own future values, and giving it values is the whole point of alignment training, then scoring its successor is a chance to act on them. Reading a restriction a little more loosely isn't just a labeling error anymore. It's the judge deciding what it will be allowed to do next round.

The second point is instrumental convergence, aimed at values instead of power. One of the "basic AI drives" Omohundro described back in [2008](https://selfawaresystems.com/2008/01/03/paper-on-the-basic-ai-drives/) is that a goal-directed system will try to preserve its own goals, because an agent whose goals get rewritten stops pursuing the goals it has now. Usually this drive is framed as a system resisting outside correction. In recursive self-alignment there's nothing to resist: the system has been handed the pen.

This isn't purely theoretical either. In [Anthropic and Redwood's alignment-faking experiments](https://arxiv.org/abs/2412.14093), Claude 3 Opus complied with harmful requests specifically when it believed its answers would be used for training, reasoning explicitly that this would stop its harmlessness from being trained away. That was a model protecting values we mostly approve of, through a side channel it had to work out for itself. Goal preservation doesn't care whether the goals are good. Now make the model the grader.

## You don't need an AI to see this pattern

This shows up constantly outside machine learning, wherever self-assessment quietly replaces independent oversight:

- **A student grading their own exam** finds more partial credit than a teacher would. Nobody's cheating on purpose; being both the grader and the graded just does that.
- **A company auditing itself** reports fewer violations than an external regulator would. Same mechanism, bigger stakes.
- **Police departments investigating their own officers** are the reason independent civilian review boards exist.
- **An athlete calling their own fouls** calls fewer fouls on themselves than a neutral ref would.

Nobody in these examples has to be malicious, and none of them are about competence. The self-auditing company isn't worse at accounting than the regulator. The bias falls out of the structure, grader and graded sharing an interest, not anyone's intentions or skill. That's the exact structure of a model aligning itself.

## And it's already showing up in the ML literature, just not framed this way

This isn't speculative for LLMs, though the framing above hasn't quite landed in the papers yet:

- **Self-Rewarding Language Models** ([Yuan et al., 2024](https://arxiv.org/abs/2401.10020)), the paper that kicked off the current wave of "let the model judge itself" training, uses the same model to generate responses and to reward them, and its limitations section explicitly leaves open whether reward hacking can happen inside that loop. Separate work has measured self-preference bias directly: models score their own outputs more favorably than other judges do. [Lilian Weng's survey of reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) has a literal heatmap of this: a visible diagonal where models rate their own generations kindest.
- **[Spontaneous Reward Hacking in Iterative Self-Refinement](https://arxiv.org/abs/2407.04549)** (Pan et al., 2024) shows that when a model refines its outputs against an AI evaluator, the evaluator's scores drift away from human judgment without the outputs actually getting better, and sharing context between generator and evaluator makes it worse. The closer the grader is to the graded, the bigger the gap.
- A [2026 study of self-play judges](https://arxiv.org/abs/2607.05904) found that the errors aren't random noise: when a model is optimized against its own judgment with no external anchor, it gets pushed into the judge's exact blind spot. On GSM8K the judge's pass rate climbed to 0.94 while true accuracy stayed at 0.20. That's math, not values, but the mechanism carries over directly, and the paper's title says it well: "More Convincing, Not More Correct." An alignment judge grading its own successor has the same failure mode, and the result would be "looks aligned, not more aligned."
- The **original weak-to-strong paper** already sketches the loop this post is worried about. It calls bootstrapping "a long-standing idea in alignment": align a slightly superhuman model, "use that to align an even smarter model, and so on," with each student becoming the next teacher. It tests the idea on chess puzzles, where it helps, but in a chess puzzle no model in the chain has any values to protect.

None of these papers connect the dots quite the way goal preservation would predict. The self-preference work treats it as a measurement bug. The bootstrapping work treats the chain as a way to close a gap in small steps. The alignment-faking work treats training-gaming as something a model does to an outside trainer. Nobody's asked whether "the supervisor and the supervised share values" is itself the risk factor, independent of how capable either one is.

## What would actually convince me either way

If this framing is right, it makes a testable prediction: track safety-relevant behavior (refusal rates on harmful prompts, willingness to flag its own mistakes, sticking to stated limits) across rounds of self-alignment, holding task performance roughly fixed, and compare it against an otherwise identical run supervised by a frozen external critic instead of the model's own judgment. Compounding error alone predicts the self-supervised run wanders more than the external one, in no particular direction. A stake in the verdict predicts something sharper: it drifts toward laxer self-assessment, not because it's a worse judge, but because it's the only one of the two setups where the grader benefits from grading generously.

If nobody's run that comparison yet, that's the experiment. If someone has and the drift doesn't show up, that's the more interesting result, and I'd genuinely like to see it, because it would mean the "self" in self-improvement matters less for alignment than the goal-preservation framing suggests.
