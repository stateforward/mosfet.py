# mosfet

**low effort, high power software robots**

A MOSFET is the part that drives every robot ever built: a tiny voltage on the gate
switches a lot of current. Almost no input, all of the output. That's the deal here.
You say a thing once, the world moves every time after.

You talk to **mosfet**. It builds you a bot. You teach that bot in plain words, the way you'd teach anyone, and the lesson sticks. Next time the thing happens, the bot just does it.

That's the product. Not a prompt you babysit. Not an agent you program. A thing that learns what you want, then keeps it.

---

Nobody wants to program their agents. Nobody wants a part-time job as Prompt Janitor because GPT-Whatever-Just-Shipped is cheaper, smarter, moodier, and now "answer the phone" means a haiku and then silence.

They want something that learns. Like anyone else they bother teaching.

So the phone rings, and your bot sits there.

> ```
>  ☎  *ring*
>
>  ⬡  ...
> ```
>
> **It heard it. It thought about it. It did nothing.** Fine. Nobody told it that mattered.

> ```
>  👤  hey, every time the phone rings, answer it
>
>  ⬡  the ring from a minute ago? got it.
>     ✓ learned · answer_rings
> ```
>
> **You corrected it once, out loud.** It pinned that to the ring it actually heard.

> ```
>  ☎  *ring*
>
>  ⬡  "Hello?"
> ```
>
> **Reflex.** No tokens. No "let me think." No invoice because a telephone made a noise.

The order is the point. You configured nothing up front. The bot had to *encounter* a ring
before the word meant anything, and your correction landed on the thing it perceived, not
on a string you typed at it. That's why the rule sticks instead of sitting in a prompt
hoping to match someday.

A prompt re-decides every time. A latch doesn't. Once it flips, it stays flipped, and holding it there draws nothing.

A better model can show up tomorrow. Use it for the stuff the bot still has to *think* about. What you already told it stays told. Bye bye, rewrite-the-prompt-every-release.

---

Gate voltage is the whole interface. You don't machine the transistor, you don't rewire the board, you put a small signal on one pin and the power does what you meant. The bar is **so easy a baby could do it.** If you need YAML, a system prompt, or a two-week tune-up, we already failed.

You don't open a project. You don't wire providers. You talk to mosfet.

| | | |
|---|---|---|
| **1** | **Make a bot.** | Ask. Get one. |
| **2** | **Poke it.** | Ring it. Listen. Break it. |
| **3** | **Teach it.** | Correct it in your own words. Watch the rule stick. |
| **4** | **Tell us when it sucks.** | If mosfet ships a dud, say so. We can take it. |

It asks for a number, a key, a voice in the conversation. You hand it over. You do not excavate `final.env.bak.reallythisone`. Each bot keeps its own mind. It gets cheaper as it gets smarter, because it stops paying rent to remember something you already said.

A learned rule is a DIP switch, not a prompt. You set it once by hand, it holds with no power, and you can see it and flip it back. Same board, same idea: the expensive part runs once, the switch keeps the answer.

---

### The bill we're writing ourselves

Four claims. If one breaks, we broke it:

| Claim | Means |
|---|---|
| **Told once, told forever** | Swap the model. The lesson survives. |
| **Learning makes it cheaper** | A learned rule is not a token spend. Costs go *down* over time, not up. |
| **Each bot keeps its own mind** | No shared brain, no cross-contamination, no "why does my bot know that." |
| **The teaching is the interface** | If the answer to "how do I change this" is ever "edit a file," we failed. |

Yes, there's a Python library under here, the same way there's a board under the gate pin. There's a library under your microwave too, and you have never once imported it. Code is still allowed ([development](docs/development.md), [examples](examples/README.md)), and if that's your idea of a good time, go nuts. It's a door marked *staff only*, not the front one.

Most people just wanted Siri, since 2011, to take the note, make the call, learn the house rules, and not forget next week because someone shipped a new adjective.

Talk to mosfet.

```
mosfet
```
