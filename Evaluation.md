### Surprise, Entropy, Cross-Entropy, and Perplexity
Ultimately, perplexity measures how surprised a model is when generating some text, so let's start from the beginning, one term at a time. Technically, the surprise of observing an outcome, $\omega$, is calculated as the negative log of the probability of that outcome:

$$\text{surprise($\omega$)} = -log(p(\omega))$$

If you have a distribution of several possible outcomes, then the average surprise is obtained by sampling $n$ times from that distribution, calculating the total surprise, and then averaging everything. In the limit of sampling infinitely from that distribution, you will find yourself calculating this quantity:

$$\sum_{\omega} p(\omega) * (-log(p(\omega)))\ \text{or} -\sum_{\omega} p(\omega)\ log(p(\omega))$$

which is simply the entropy of that distribution. Thus, entropy is the average surprise from observing events sampled according to $p$ (the second part of the equation), if you assume that $p$ itself is the true model of reality. However, what if you want to measure surprise if you assume that $p$ is the true model of reality, but you are sampling from another distribution $q$? More correctly, $p$ already has an average surprise, qua entropy, on its own, so the correct question would be "How much **extra** surprise do you get if you assume that $p$ is the true model of reality, but you are sampling from another distribution $q$?" Welcome to cross-entropy, a quantity computed as follows:

$$-\sum_{\omega} p(\omega)\ log(q(\omega))$$

To put it in context, let's say your vocabulary consists of ten words for simplicity and that your model has already predicted three words in a sentence. 
Now it's the time to predict the fourth word. According to your training dataset, you have a probability of observing each of the ten words given the already predicted three words, which constitute the distribution $p_4 = p(W_4=w_4|W_1=w_1, W_2=w_2, W_3=w_3)$ for each $w_4$ in your vocabulary. This distribution has some average surprise $\text{entropy}(p_4)$.

Let's say that your model predicts a different distribution for the fourth word given the first three, $q_4 = q(W_4=w_4|W_1=w_1, W_2=w_2, W_3=w_3)$. The additional surprise that you get from using this model to predict the fourth word assuming that $p_4$ is the correct distribution is simply the cross-entropy between $p$ and $q$

$$-\sum_{\text{word} \in V} p(W_4=\text{word}|W_1=w_1, W_2=w_2, W_3=w_3)\ log(q(W_4=\text{word}|W_1=w_1, W_2=w_2, W_3=w_3))$$

During evaluation, each word in the vocabulary can potentially have a non-zero probability, so for each new token, you compute this sum if you wish to compute the CE. However, during training, only one word is correct, so all tokens, except the correct one, have zero probabilities, and the correct one has a probability of 1.

Taking all of this to arrive at perplexity, this metric is the exponential of the CE loss. However, the CE is not between the true (empirical) distribution and the model's predicted distribution but between a uniform one and the model's prediction. In other words, $p$ is a uniform distribution while $q$ is the model's predicted distribution. With the standard CE loss, $p$ is the empirical distribution, a delta distribution around the true word in the sequence.

From HF: "Intuitively, it can be thought of as an evaluation of the model’s ability to predict **uniformly** among the set of specified tokens in a corpus" ([link](https://huggingface.co/docs/transformers/perplexity#:~:text=Intuitively%2C%20it%20can%20be%20thought%20of%20as%20an%20evaluation%20of%20the%20model%E2%80%99s%20ability%20to%20predict%20uniformly%20among%20the%20set%20of%20specified%20tokens%20in%20a%20corpus.))

### HumanEval
This is a dataset that probes performance on code-related tasks using unit tests.

#### pass@k metric
This is the probability that at least 1 of $k$ generated solutions works for a given code problem. This is naïvely calculated by generating $k$ samples and computing it. However, this can lead to a high variance.

An alternative is to generate $n >> k$ samples and then compute an unbiased estimate of the probability that at least one sample is correct out of the $k$ samples sampled from the $n$ ones:

Here, $c$ is the number of correct solutions for a given task. Note that the average is over multiple problems/tasks.

$$
\text { pass } @ k:=\underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{\left(\begin{array}{c}
n-c \\
k
\end{array}\right)}{\left(\begin{array}{c}
n \\
k
\end{array}\right)}\right]
$$

#### Nucleus sampling
Nucleus sampling is a variant of top-p sampling, where instead of sampling from the top-p most likely tokens, the model samples from the smallest possible set of tokens whose cumulative probability exceeds a 
certain threshold p. This threshold is called the nucleus, and it is dynamically determined based on the probability distribution of the tokens at each step of the generation process.

The steps are as follows:

- get the raw probabilities
- obtain the set of tokens whose cumulative probability exceeds a certain threshold p, called the nucleus
- rescale the probabilities of the surviving tokens by normalisation
- sample based on the new distribution

Example:
todo: add a numerical example

### LLM Benchmarks
