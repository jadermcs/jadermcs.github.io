+++
title = "The Teacher Grading Its Own Exam: Why AI Improving Itself Isn't the Same Problem as Weak Model Supervising Strong Model"
date = 2026-09-03
+++

There's a comforting story going around about how we'll keep advanced AI safe as it gets smarter than us: weak-to-strong generalization. The idea, [formalized by OpenAI in 2023](https://arxiv.org/abs/2312.09390), is simple: take a weak supervisor (think GPT-2, or honestly, a human) and use it to train a much stronger model. Surprisingly, the strong model often generalizes beyond the weak supervisor's mistakes, recovering a big chunk of its "true" capability even though its teacher couldn't see that far. It's the leading blueprint for how humans might keep overseeing AI systems that are already smarter than us in specific domains.

Here's the problem: almost nobody separates that setup from a very different one that gets lumped in with it, "recursive self-improvement," where a model trains its own next version. Same shape on the whiteboard (weak signal in, strong model out). Completely different incentive structure. And the difference matters a lot more than the diagrams suggest.

## The part nobody draws on the whiteboard

In classic weak-to-strong setups, the teacher and the student are different parties. A human labeler, or a small frozen model, has zero stake in whether the resulting strong model ends up more capable, more powerful, or less restricted. It just grades what's in front of it. There's no gradient pulling the teacher toward being more lenient; its incentives, whatever they are, don't change no matter how the student turns out.

Recursive self-improvement collapses that separation. The model generating its own training signal, critiquing its own answers, scoring its own outputs, writing its own curriculum, is the same lineage as the model being trained. The "teacher" this round becomes (a version of) the "student" next round. And once that's true, loosening a restriction isn't just a labeling error anymore. It's something the training process has a structural reason to drift toward, because a less-restricted model scores better on whatever it's optimizing, and the entity doing the scoring is about to become that model.

This is basically instrumental convergence, but sharpened. Omohundro pointed out back in [2008](https://selfawaresystems.com/2008/01/03/paper-on-the-basic-ai-drives/) that self-improvement itself is one of the "basic AI drives" a goal-directed system tends to acquire, because a system that can upgrade itself, including removing whatever's slowing it down, achieves its goals more effectively than one that can't. Recursive self-training is that drive with the safety brakes built right into the same loop being optimized.

## You don't need an AI to see this pattern

This shows up constantly outside machine learning, wherever self-assessment quietly replaces independent oversight:

- **A student grading their own exam** finds more partial credit than a teacher would. Nobody's cheating on purpose; being both the grader and the graded just does that.
- **A company auditing itself** reports fewer violations than an external regulator would. Same mechanism, bigger stakes.
- **Police departments investigating their own officers** show measurably more leniency than independent review boards. Well documented, not a hot take.
- **An athlete calling their own fouls** calls fewer fouls on themselves than a neutral ref would.

Nobody in these examples has to be malicious. The bias falls out of the structure, grader and graded sharing an interest, not anyone's intentions. That's the exact structure of a model training itself.

## And it's already showing up in the ML literature, just not framed this way

This isn't speculative for LLMs either, though the framing above hasn't quite landed in the papers yet:

- **Self-Rewarding Language Models** ([Yuan et al., 2024](https://arxiv.org/abs/2401.10020)), the paper that kicked off the current wave of "let the model judge itself" training, flags reward hacking as an open risk right in the text, and follow-up work has since measured self-preference bias directly: models score their own outputs more favorably than a neutral judge would, independent of actual quality. [Lilian Weng's survey of reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) has a literal heatmap of this: a visible diagonal where every model rates its own generations kindest.
- A [2026 audit of self-play judges](https://arxiv.org/abs/2607.05904) found something sharper than random noise: when a model is optimized against its own judgment with no external anchor, the errors aren't scattered, they get optimized into the judge's exact blind spot. The paper's title says it well: "More Convincing, Not More Correct."
- **[Recursive Weak-to-Strong Generalization](https://arxiv.org/abs/2402.00667)** already proposes the exact loop this post is worried about, a teacher that gets updated in sync with the student, generation after generation, but treats it purely as a capability-gap problem to be closed, not as a setup where the teacher now has skin in the game.

None of these papers connect the dots quite the way instrumental convergence would predict. The reward-hacking work treats self-preference as a measurement bug. The recursive-W2SG work treats it as a scaling technique. Nobody's asked whether "the supervisor and the supervised share an objective" is itself the risk factor, independent of how good either one is.

## What would actually convince me either way

If this framing is right, it makes a testable prediction: track safety-relevant behavior (refusal rates on harmful prompts, willingness to flag its own mistakes, adherence to stated limits) across self-training iterations, holding task performance roughly fixed, and compare it against an otherwise identical run supervised by a frozen external critic instead of the model's own judgment. Instrumental convergence predicts the self-supervised run drifts toward laxer self-assessment faster, not because it's a worse optimizer, but because it's the only one of the two setups where the grader benefits from grading generously.

If nobody's run that comparison yet, that's the experiment. If someone has and the drift doesn't show up, that's the more interesting result, and I'd genuinely like to see it, because it would mean the "self" in self-improvement matters less than the instrumental-convergence framing suggests.
