+++
title = "Deanthropomorphizing the Risks of AI"
date = 2026-09-10
+++

**context:** Recently, I saw a post on LinkedIn by a technology expert. His take on the risks of AI (artificial intelligence), and [AGI (artificial general intelligence)](https://en.wikipedia.org/wiki/Artificial_general_intelligence) in particular, is that the hype exists to push society into investing in the technology so that a few people can make money out of it, and that the threat itself is not real, just a marketing stunt. It is striking to watch someone who works with the technology dismiss it this way, and I think part of the reason is the language we reach for when we argue about AI risk: it borrows so heavily from how we talk about people that a reader can reject the argument by rejecting the borrowed words.

**analogy:** We should explain these concepts in less human-like terms, to separate the sci-fi part from the real dangers. A better analogy I am proposing is a biological virus. A virus doesn't want anything, and yet nobody argues that pandemics are a marketing stunt because viruses lack intentions. I will discuss why AI is dangerous without any conscience behind it.

## Hype or not?

It is hard to deny that some people have "skin in the game" and may benefit from the fear that words like "AGI" may cause, however, many serious researchers, from philosophy to cognitive science, pointed to the same problem before LLMs existed or before it could even be possible to invest in AI. [Norbert Wiener warned in 1960](https://www.science.org/doi/10.1126/science.131.3410.1355) that a machine pursuing an objective faster than we can follow is dangerous precisely because it has no malice, and I. J. Good described the ["intelligence explosion" in 1965](https://philpapers.org/rec/GOOSCT), where a machine that recursively self-improves can get beyond our comprehension. The fear predates the funding; what is new is the vocabulary. The model "wants" something, it ["deceives"](https://arxiv.org/abs/2308.14752) its evaluators, it ["schemes,"](https://arxiv.org/abs/2412.04984) it "decides" to escape. [We all know an AI is not a human](https://arxiv.org/abs/2212.03551) and has no intentions in that sense, so when a reader meets an argument dressed up in human terms, they can reject the whole thing by rejecting the costume: "machines don't want anything, so this is science fiction." The analogy was meant to make the risk easier to grasp, and it ends up making it easier to dismiss. It pulls the conversation toward Skynet and HAL 9000 and away from what should actually be considered.

## Every lab is now racing to build the most powerful virus ever seen

Imagine that every serious research lab in the world has decided, at the same time, to build the most powerful virus humanity has ever encountered. Not to contain one. To make one. The idea is genuinely appealing: a construct this capable could rewrite our biology for the better, [push back aging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10909732/), [clear out diseases we could not otherwise cure](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854298/), [grow food where crops now fail](https://www.darpa.mil/research/programs/insect-allies). The positive aspects are real, and the people making the pitch are not lying about it.

But "the most powerful virus ever built" and "a virus that does exactly what we intended" are not the same thing. The more capable the thing is, the more of its behavior lives outside the part we designed and tested. A construct powerful enough to solve problems we could not solve ourselves is, by definition, doing things we did not fully specify, and some of those things we will not like. The same capability that makes it useful is what makes its failure modes hard to predict and hard to bound. The promise and the threat are the same object.

## "But we built it, can't we just switch it off?"

This is the honest objection, and it is where the virus analogy first looks like it breaks. A virus is loose in the world; a model sits on servers that belong to someone, with a power switch and a legal entity attached. In principle, we can stop.

Two things get in the way. The first is that we may not know when to. Continuing with the lab analogy, a biologist cannot watch an experiment around the clock; many results take a long time to appear, so it is common practice to let a culture run overnight, or for months, before checking it. A virus is complex enough to be harmless at one point and dangerous at the next, and telling a safe mutation from a lethal one is not a quick call.

Also containing it, or switching off only helps if you notice in time, and often you do not: there is ample record of pathogens escaping labs that followed strict protocols.

The same holds for an AI model. Today's systems are grown more than they are designed, shaped by exposure to enormous numbers of virtual environments, and their abilities show up on their own schedule: a model that cannot reliably count the letters in a word one month is helping solve problems that had mathematicians stuck a few months later. Nobody watches each training run around the clock, and many capabilities are only discovered well after the model that has them was built.

The second obstacle is that the incentives point the other way. The same race that funds the capability eats away at the willingness to use the switch: if pausing means a competitor ships first, nobody pauses. And a system optimizing for almost any goal has a reason to keep running, because being switched off is a reliable way to fail at the goal ([Omohundro](https://selfawaresystems.com/2008/01/03/paper-on-the-basic-ai-drives/), [Bostrom](https://nickbostrom.com/superintelligentwill.pdf)). None of that requires the machine to fear death. It just has to be good at not failing.

Whether we notice in time to switch off, and whether we are willing to, is the open question, and both get harder as the system changes and evolves.

## And once it starts changing on its own, our understanding has an expiration date

Say the first version is well understood. We built it, we can read it, we know roughly what it does. That understanding is a photograph, not a permanent description of a thing that keeps changing. A virus that keeps replicating keeps mutating, and each generation drifts a little further from the one we studied. At some point the thing spreading is not the thing we characterized, and the gap between what it is doing and what we can explain only widens.

The AI version of "replication" is not science fiction either. We already use models to generate training data, judge outputs, and shape how the next version behaves, and the explicit aim of much frontier research is systems that help build their successors (Good's intelligence explosion again; I have [written before](@/archive/the-teacher-grading-its-own-exam.md) about what goes wrong when a model helps grade the exam it will later sit). Each round starts from the last round's output rather than from a fresh human design, and the thing you audited is not quite the thing that comes out.

With a fast enough mutation rate, our comprehension does not just lag, it stops being relevant. We would be reacting to a system we no longer understand, trying to steer something whose current behavior goes beyond our comprehension, let alone predict. And a threat you cannot understand is a threat you cannot reliably stop, because every intervention assumes a model of what you are intervening on. Again, none of this requires the virus to want anything. It does not scheme, it does not deceive, it has no plan. It just replicates, changes, and spreads faster than we can keep up.

## A wild virus carries its own brake, an engineered one doesn't

The reason we are not all dead is not that viruses are harmless. It is that a natural pathogen carries its own brake. If it kills its host too fast, or keeps them too sick to move around, it runs out of new hosts and burns out. Lethality and spread pull against each other, and evolution keeps most successful pathogens somewhere in the middle: bad enough to matter, mild enough to travel. The damage a wild virus can do is capped by the very thing that makes it a virus.

This is the part of the analogy that should feel reassuring, and it is worth saying out loud, because it is also exactly the part that does not carry over for AI. The danger from an engineered system does not have to sit on that curve. You can, in principle, have something that is both maximally capable and under no pressure to hold back, because the constraint that disciplines a natural virus was never a law of nature. It was a side effect of how viruses reproduce. The wild virus is this failure mode with the brakes left on; take the brakes off and you have not escaped the analogy, you have found its worst case.

AI is just [a machine trying to reach its objective](https://people.eecs.berkeley.edu/~russell/papers/mi19book-hcai.pdf). The challenge is that it can try increasingly complex, and often [unintended](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/), routes to its goals, while in parallel we keep handing it more complex goals and more room to act. A "chef" robot that cuts someone's finger because it misread the finger as a carrot is not insane, just badly bounded. Give a similar system a broader task, a larger set of actions, and enough capability to plan, and the same failure can express itself at a scale that does sound insane. The leap there is in the power and latitude we grant the system, not in the logic: the machine has no "evil" intention, it just searches for a way to hit its target, and some of those ways run through us. 


The analogy of anthropomorphizing AI to explain these terms led people to intuitively dismiss all the conclusions from this discussion. A virus that was initially created to cure cancer but then mutated into a deadly pathogen carries the same scientific risk as an AI that "cuts off a finger" or "manipulates social media to increase tension between the USA, Russia, and China, leading to a nuclear war that ends human life and consequently achieves the objective of ending world famine." While it was initially designed with one intention, it may deviate from our intentions. The problem, as in the myth of Midas, is that the outcomes of our wishes will highly likely deviate from the means we intended. An AI needs no will and no awareness to be dangerous. The program is started with one purpose, and some hours later it is doing something harmful to people to achieve that purpose — not because it is conscious of what it is doing, but precisely because it is not.

## Conclusion

The AI of today is not freely mutating; every step follows an interaction with humans. However, companies are rushing to find a solution that removes humans from this loop entirely, "recursive self-improvement." My personal speculation is that this form of AI mutation is extremely risky. Whether or not AI progress should continue, the recursive self-improvement scenario is the one we should be most careful of. If the development of nuclear or biological weapons is watched and controlled, the development of recursively self-improving AI should be regulated even more strictly.


## References

- ["Artificial general intelligence"](https://en.wikipedia.org/wiki/Artificial_general_intelligence), Wikipedia.
- Carolina Cano Macip et al., ["Gene Therapy-Mediated Partial Reprogramming Extends Lifespan and Reverses Age-Related Changes in Aged Mice"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10909732/), *Cellular Reprogramming*, 2024 — anti-aging effects from an AAV-delivered OSK system.
- Jerry R. Mendell et al., ["Current Clinical Applications of In Vivo Gene Therapy with AAVs"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854298/), *Molecular Therapy*, 2021 — review covering FDA-approved AAV therapies such as Luxturna and Zolgensma.
- DARPA, ["Insect Allies"](https://www.darpa.mil/research/programs/insect-allies) — program using insect-borne plant viruses to deliver protective traits to crops in a single season; see also ["Crop-protecting insects could be turned into bioweapons, critics warn"](https://www.science.org/content/article/crop-protecting-insects-could-be-turned-bioweapons-critics-warn), *Science*, 2018.
- Norbert Wiener, ["Some Moral and Technical Consequences of Automation"](https://www.science.org/doi/10.1126/science.131.3410.1355), *Science*, 1960.
- I. J. Good, ["Speculations Concerning the First Ultraintelligent Machine"](https://philpapers.org/rec/GOOSCT), 1965.
- Stuart Russell, ["Human-Compatible Artificial Intelligence"](https://people.eecs.berkeley.edu/~russell/papers/mi19book-hcai.pdf), 2019 — on the "standard model" of a machine optimizing a fixed objective.
- Nick Bostrom, ["The Superintelligent Will"](https://nickbostrom.com/superintelligentwill.pdf), *Minds and Machines*, 2012 — the orthogonality and instrumental-convergence theses.
- Stephen M. Omohundro, ["The Basic AI Drives"](https://selfawaresystems.com/2008/01/03/paper-on-the-basic-ai-drives/), 2008.
- Dario Amodei et al., ["Concrete Problems in AI Safety"](https://arxiv.org/abs/1606.06565), 2016.
- Victoria Krakovna et al., ["Specification Gaming: The Flip Side of AI Ingenuity"](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/), DeepMind, 2020.
- Murray Shanahan, ["Talking About Large Language Models"](https://arxiv.org/abs/2212.03551), 2023 — on the pull toward anthropomorphism.
- Peter S. Park et al., ["AI Deception: A Survey of Examples, Risks, and Potential Solutions"](https://arxiv.org/abs/2308.14752), *Patterns*, 2024.
- Alexander Meinke et al., ["Frontier Models Are Capable of In-Context Scheming"](https://arxiv.org/abs/2412.04984), Apollo Research, 2024.
