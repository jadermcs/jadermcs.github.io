+++
title = "The Nurse Is Worth $160: On Confusing a Price Tag With a Pile of Stuff"
date = 2026-09-04
+++

Take a nurse. Break her down into elements and price the result: oxygen, carbon, hydrogen, nitrogen, some calcium and phosphorus, trace iron. [Estimates vary](https://www.sciencefocus.com/the-human-body/how-much-money-is-a-human-body) with how you count — about $160 at commodity prices, up to roughly $150,000 if you insist on pharmaceutical purity. Call it $160. That is her material value: everything she is made of, at market.

Now ask what she is worth over the next ten years. At the US median registered-nurse wage of [$97,550](https://www.bls.gov/ooh/healthcare/registered-nurses.htm), a decade of work discounted at 5% is about $750,000 in present value — and that's a floor, not a ceiling, because wages capture only the slice she can bargain for. The patients who don't die, the errors caught at 3am that never become incidents: none of it reaches her paycheck. Price avoided deaths the way regulators do, at a [value of a statistical life](https://www.epa.gov/environmental-economics/mortality-risk-valuation) of $7-10 million, and one prevented death in a decade dwarfs her entire salary line.

Nobody defends the $160. The material composition is obviously, insultingly, the wrong frame.

And yet it is exactly the frame a lot of public argument about fortunes runs on.

## Stock, flow, and the thing in between

Two different things get called "value." The first is what something is *made of* — scrap value, liquidation value, the pile you're left holding if you take it apart today. The nurse's $160. The second is what it's expected to *produce* — the stream of future output, discounted, adjusted for how likely that output is to arrive. The nurse's $750,000.

Finance calls the second one discounted cash flow: project what an asset generates, discount it because a dollar in year nine is worth less than a dollar today, sum ([Damodaran's notes](https://pages.stern.nyu.edu/~adamodar/pdfiles/eqnotes/dcfallOld.pdf) if you want the machinery). The structural fact is that a DCF valuation is not a description of a hoard. It's a **forecast with a price attached**.

Applied to people, this isn't a metaphor. Gary Becker's [*Human Capital*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1496221) (1964) is built on exactly it: a person's economic value as the discounted present value of their future earnings, with education as an *investment* because you pay now for a larger stream later. There are [published tables](https://escholarship.org/content/qt82d0550k/qt82d0550k.pdf?t=li5h7t) of lifetime earnings by age and sex — it's how courts put a number on a wrongful death. The nurse's potential value is her human capital, and it runs about four thousand times her scrap value.

## Where the risk lives

The $750,000 isn't a fact about the nurse. It's a *distribution* over futures involving her, collapsed into one number. She might burn out at year three, get injured, or become a nurse practitioner and double the stream. She carries what the [lifecycle finance literature](https://www.researchgate.net/publication/251852872_Lifetime_Financial_Advice_Human_Capital_Asset_Allocation_and_Insurance) calls a unique mortality risk: loss of the entire stream at once.

That literature treats human capital as an asset class with a risk profile, which is a useful lens. Some streams are bond-like — a tenured professor's looks like an inflation-linked bond. Some are stock-like and volatile: a founder, a commissioned salesperson, an athlete. All of them are deeply illiquid, since you can't sell a claim on your own future labor, and the fact that you can't is a legal choice rather than an economic necessity.

Risk enters through the discount rate. Riskier streams get discounted harder, so a racing driver's earnings and a nurse's, identical on paper, aren't worth the same present value. But underneath risk sits something worse, which Frank Knight named in [*Risk, Uncertainty, and Profit*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1496192) (1921): risk is when you can put probabilities on outcomes, uncertainty is when you can't. Nobody had a probability distribution over "smartphones" in 2005. Knight's point was that profit is the *residual* left after every fixed claim is paid — the compensation for bearing exactly that unmeasurable uncertainty. An entrepreneurial fortune is, structurally, the accumulated payoff of a bet that could have gone to zero.

## The fortune-as-a-hoard mistake

Now put the halves together. When a net worth is reported as $80 billion, that number is almost never a pile of anything. It's equity: shares priced by a market, and the price is itself a discounted forecast of future profits. [Paper wealth](https://en.wikipedia.org/wiki/Paper_wealth) is the dry term — what the assets *could* sell for, at current prices, assuming the sale doesn't move the price. Which for a controlling stake it emphatically would.

So the $80 billion is the same *kind* of number as the nurse's $750,000. It is not $80 billion of stuff sitting somewhere.

Treating it as a hoard produces a specific error: assuming the quantity is fixed and merely *located* in the wrong hands, so that moving it changes who has it but not how much there is. That's the [fixed pie fallacy](https://en.wikipedia.org/wiki/Fixed_pie_fallacy) with a valuation twist, and the twist catches people who'd never fall for the plain version. The plain mistake is thinking total wealth is constant. The subtle one is thinking *a particular measured fortune* is a constant that survives being moved. It often isn't — the same way you can't harvest the nurse's $750,000 by liquidating the nurse. What you'd get is $160.

## The chicken and the egg

Underneath the measurement problem sits a worse one, and it's the most interesting part of the question.

Suppose every implementation objection is solved: valuation methods for illiquid stakes, a withholding mechanism, enforcement good enough to stop the obvious dodges. Grant all of it. Something circular is still sitting at the bottom.

A price is never a fact about an asset alone. It's a fact about an asset *in a configuration*. The $80 billion is quoted for a world in which that stake stays concentrated, claims like it trade freely at prices like that, and the stream being discounted is expected to reach whoever holds the claim. Those aren't background details the valuation abstracts away from; they're the conditions under which the quote means anything. So asking what the stake is worth after a large redistribution isn't asking about the same number in a new location. It's asking for a different quote, under a configuration nobody has observed, using a price discovered under the one we have.

The mild version of this is already measured: a credible tax on future returns doesn't wait for collection day, it gets [capitalized into the price immediately](https://www.nber.org/system/files/working_papers/w12342/w12342.pdf), because buyers discount what they'll pay to compensate for tax they'll owe later. The base shrinks partly because you announced you were coming for it. The general version has a name — the [Lucas critique](https://www.federalreserve.gov/econres/ifdp/post-econometric-policy-evaluation-a-critique.htm) (1976): you can't use a relationship estimated under one policy regime to predict outcomes under another, because the quantities you measured were chosen by forward-looking people responding to the regime you're about to change. The $80 billion is a parameter estimated under the current rules.

That gives the circularity its shape. The valuation is evidence about a world in which the valuation is left alone. The size of the prize is partly a measurement of the arrangement that produced it — including, uncomfortably, that arrangement's not redistributing.

I'll stop short of where this argument wants to go, because taken to its end it proves far too much. *Every* tax changes its own base; income taxes reduce labor supply somewhat and we tax income anyway. Endogeneity doesn't veto anything. It converts a yes/no question into a magnitude question: how much survives the repricing and the restructuring, and is the remainder worth collecting? That has a real answer, and it differs for every design.

## What the analogy doesn't prove

Three ways this gets pushed into a claim it doesn't support.

**Earnings aren't social contribution.** DCF prices the stream *captured*, not the value *created*, and those come apart badly under market power, regulatory capture, and lucky positioning. "Fortune = measured potential to create value" holds only where prices track social value, which is an assumption, not a theorem. Note which way it cuts: it says the nurse is underpriced, not that the framing is wrong.

**"Only paper wealth" proves less than it seems.** Illiquid assets still buy real things — collateral, control over where capital gets deployed, influence over the rules. You can accept every word about stocks versus flows and still think the flows should be taxed differently.

**Forecasts inflate.** If a valuation is a forecast, it can be a bubble, and then the honest gloss on "a measurement of expected future production" is "a measurement of what people currently believe about future production."

## No problem this size is easy

Put it together and the picture is symmetric, which is why nobody likes it.

"Take the $80 billion and give it to people who need it" sounds like one action. It's at least four, and each one leaks. **Valuing it** rests on a price conditional on rules you're about to change. **Converting it** needs someone on the other side of the trade — at scale, pension funds and other large holders — so a big redistribution is partly a reshuffle of who holds the claim rather than paper becoming groceries. **Landing it** on the intended person runs into tax incidence, where who writes the check and who bears the cost have been a century-long research question. **Keeping the stream alive** afterward depends on behavioral response: not "the rich will all flee," which is a talking point the evidence doesn't support at that magnitude, but not zero either.

There's also a composition problem, and it's the one with the longest shadow. That $80 billion isn't sitting in a warehouse of yachts. By Altrata's 2024 Billionaire Census, business equity is about 66% of billionaire wealth, liquid assets 31%, and real estate and luxury goods 3% (I found the figures [via Cato](https://www.cato.org/blog/billionaires), whose framing you should weigh separately from the count). What the number mostly *is* is ownership claims on operating firms: payroll, equipment, plants, R&D budgets. Productive capital, in the plain sense.

Which means converting it into current consumption is an intertemporal trade, not a free lunch. This is the oldest result in growth economics — the [golden rule](https://en.wikipedia.org/wiki/Golden_Rule_savings_rate) formalizes the tradeoff, but the intuition is older than the model and known as eating the seed corn. Distribute the capital stock as food this year and there is more food this year, and less of whatever that capital would have built over the next thirty. The people harmed by that are mostly not yet in the room to object.

Two things keep this from being a trump card, though. First, capital doesn't evaporate when ownership changes hands — factories don't stop existing because the share register did. The risk is *reallocation from investment to consumption*, and whoever ends up directing the capital may do it better or worse than the incumbent. That's an empirical question about allocation quality, not a law of nature. Second, and this is where the essay eats its own tail: transfers aren't automatically consumption. Money moved into nursing schools, clinics, and childhood health is capital formation — human capital, by the exact accounting that got the nurse from $160 to $750,000 in the first place. Moving a claim off a share register and into a nursing program isn't eating the seed corn. It's planting different seed, in a crop we happen to measure worse. Whether it's a good trade depends on the relative returns, which is, again, a magnitude nobody gets to assert.

The other side leaks just as badly. "It's only paper wealth, so nothing can be done" is the same error inverted — treating a forecast as untouchable rather than as a hoard, mistaking *hard to value* for *must not be touched*. The nurse cuts against complacency more than she cuts for it: if her captured stream is a fraction of what she creates, that gap is a real distortion, and noticing that fortunes are forecasts doesn't explain it away.

So the honest position is uncomfortable from both directions. A measured fortune is not a pile you can pick up and carry. It is also not a mirage you're obliged to ignore. What's left is unglamorous — tax design, incidence estimates, valuation rules for illiquid assets, second-order effects, and a great deal of arguing about magnitudes. That's why the serious version of this debate reads nothing like the slogans: Saez and Zucman's [progressive wealth tax](https://eml.berkeley.edu/~saez/saez-zucmanBPEAoct19.pdf) and [capital-gains withholding proposal](https://gabriel-zucman.eu/files/SYZ2021.pdf) are interesting precisely because they try to engineer around specific leaks instead of wishing them away, and the [published critiques](https://www.columbia.edu/~wk2110/bin/BPEASaezZucman.pdf) are worth as much as the proposals.

Big problems don't come with moves that are simultaneously simple, large, and free. The nurse doesn't tell you what to do about billionaires. She tells you that anyone certain of the answer is working from the wrong number — because the reason she's worth more than her calcium is the same reason a fortune isn't a stack of gold in a room. Both are claims on futures that might not arrive, priced under uncertainty, by people who don't know how it ends.
