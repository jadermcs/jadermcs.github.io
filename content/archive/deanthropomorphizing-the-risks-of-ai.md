+++
title = "Deanthropomorphizing the Risks of AI"
date = 2026-09-10
+++

**context:** Recently, I saw a post on LinkedIn by a technology expert. His take on the risks of AI (artificial intelligence), and [AGI (artificial general intelligence)](https://en.wikipedia.org/wiki/Artificial_general_intelligence) in particular, is that this overhypes society into investing in this new technology so that people can make money out of it, concluding that the threat is not real, just a marketing stunt. Even technology experts diminish the risk of this technology, and I believe the fault lies in the fact that, for the sake of analogy, we anthropomorphize many of the concepts we use to argue about AI. This is a problem because we know that AI is not human and has no intention, but because of this analogy, the reader may invalidate or undervalue the entire argument. I think that this anthropomorphic analogy creates more of a sci-fi view of the problem than what should actually be considered.

**analogy:** We should explain these concepts in less human-like terms, to separate the sci-fi part from the real dangers. A better analogy is a biological virus. A virus doesn't want anything, and yet nobody argues that pandemics are a marketing stunt because viruses lack intentions.

## Hype or not?

It is hard to deny that some people have "skin in the game" and may benefit from the fear that words like "AGI" may cause, however, many serious researchers, from philosophy to cognitive science, pointed to the same problem before LLMs existed or before it could even be possible to invest in AI. [Norbert Wiener warned in 1960](https://www.science.org/doi/10.1126/science.131.3410.1355) that a machine pursuing an objective faster than we can follow is dangerous precisely because it has no malice, and I. J. Good described the ["intelligence explosion" in 1965](https://philpapers.org/rec/GOOSCT). I think a big part of the fault is that, for the sake of analogy, we anthropomorphize most of the concepts we use to talk about AI risk. The model "wants" something, it ["deceives"](https://arxiv.org/abs/2308.14752) its evaluators, it ["schemes,"](https://arxiv.org/abs/2412.04984) it "decides" to escape. [We all know an AI is not a human](https://arxiv.org/abs/2212.03551) and has no intentions in that sense, so when a reader meets an argument dressed up in human terms, they can reject the whole thing by rejecting the costume: "machines don't want anything, so this is science fiction." The analogy was meant to make the risk easier to grasp, and it ends up making it easier to dismiss. It pulls the conversation toward Skynet and HAL 9000 and away from what should actually be considered.

## Every lab is now racing to build the most powerful virus ever seen

Imagine that every serious research lab in the world has decided, at the same time, to build the most powerful virus humanity has ever encountered. Not to contain one. To make one. The pitch is genuinely appealing: a construct this capable could rewrite our biology for the better, [push back aging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10909732/), [clear out diseases we could not otherwise cure](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854298/), [grow food where crops now fail](https://www.darpa.mil/research/programs/insect-allies). The upside is real, and the people making the pitch are not lying about it.

But "the most powerful virus ever built" and "a virus that does exactly what we intended" are not the same sentence. The more capable the thing is, the more of its behavior lives outside the part we designed and tested. A construct powerful enough to solve problems we could not solve ourselves is, by definition, doing things we did not fully specify, and some of those things we will not like. The same capability that makes it useful is what makes its failure modes hard to predict and hard to bound. The promise and the threat are the same object.

## And once it starts changing on its own, our understanding has an expiration date

Say the first version is well understood. We built it, we can read it, we know roughly what it does. That understanding is a photography, not a permanent description of a thing that is constantly changing. A virus that keeps replicating keeps mutating, and each generation drifts a little further from the one we studied. At some point the thing spreading is not the thing we characterized, and the gap between what it is doing and what we can explain only widens.

With a fast enough mutation rate, our comprehension does not just lag, it stops being relevant. We would be reacting to a system we no longer understand, trying to steer something whose current behavior we cannot describe, let alone predict. And a threat you cannot understand is a threat you cannot reliably stop, because every intervention assumes a model of what you are intervening on.

None of this requires the virus to want anything. It does not scheme, it does not deceive, it has no plan. It just replicates, changes, and spreads faster than we can keep up, driven by [instrumental goals](https://nickbostrom.com/superintelligentwill.pdf). That is the shape of the risk worth arguing about, and you can state the whole thing without a single human verb.

## We know virus bounds, but we don't know about AI

The reason we are not all dead is not that viruses are harmless. It is that a natural pathogen carries its own brake. If it kills its host too fast, or keeps them too sick to move around, it runs out of new hosts and burns out. Lethality and spread pull against each other, and evolution keeps most successful pathogens somewhere in the middle: bad enough to matter, mild enough to travel. The damage a wild virus can do is capped by the very thing that makes it a virus.

This is the part of the analogy that should feel reassuring, and it is worth saying out loud, because it is also exactly the part that does not carry over for AI. The danger from an engineered system does not have to sit on that curve. You can, in principle, have something that is both maximally capable and under no pressure to hold back, because the constraint that disciplines a natural virus was never a law of nature. It was a side effect of how viruses reproduce.

AI is just [a machine trying to reach its objective](https://people.eecs.berkeley.edu/~russell/papers/mi19book-hcai.pdf). The challenge and threat that AI exposes is that it can try increasingly complex, and often [unintended](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/), ways to arrive at its goals, and in parallel we are giving it more complex goals. For instance, if a "chef" robot cuts someone's finger because it misidentified the finger as a carrot, that doesn't sound insane. But if this chef robot now has a more complex task and figures out that bombing some place may be the best way to solve a problem, it is the same nature of threat: the machine has no "evil" intention, it will just find a way to accomplish its objective, and that way may hurt humans, whether by cutting fingers or bombing entire countries.

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
