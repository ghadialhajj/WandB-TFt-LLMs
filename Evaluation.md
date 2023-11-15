### Perplexity
Perplexity measures how surprised a model is when generating some text. Technically, surprise is calculated as the negative log probability.

### HumanEval
This is a dataset that probes performance on code-related tasks using unit tests.

#### pass@k metric
This is the probability that at least 1 of $k$ generated solutions works for a given code problem. This is naïvely calculated by generating $k$ samples and computing it, however, this can lead to a high variance.

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
