---
title: "Naive Bayes: the not-so-naive explanation"
excerpt_separator: "<!--more-->"
categories:
  - Generative Models
tags:
---

(Introduction to Naive Bayes: generative model, naive Bayes assumption)

(Spam email problem)

(Multinomial way to encode emails)

We are going to model $ p(x_j \vert y) $ after a multinomial distrubution. Let's parameterize the distribution with $ \phi_y = p(y=1)$, $ \phi_{k \vert y=1} = p(x_j = k \vert y = 1) $ and $ \phi_{k \vert y=0} = p(x_j = k \vert y = 0) $. Given this parameterization, we assume that $ p(x_j = k \vert y) $ is the same for every j, i.e. the distribution after which a word is generated is independent of the position of that word in the email.

Given a dataset of emails $ \{(x^{(i)}, y^{(i)}); i = 1,...,m\} $ with $ x^{(i)} \in \mathbb{R^{d_i}} $ where $ d_i $ is length of $ i $-th email, the likelihood of the dataset is

$$
\begin{align}
L(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) 
&= \prod_{i=1}^{n} p(x^{(i)}, y^{(i)}) \\ 
&= \prod_{i=1}^{n} \left( \prod_{j=1}^{d_i} p(x_{j}^{(i)} \vert y^{(i)}; \phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) \right) p(y^{(i)})
\end{align}
$$

Taking $ log $ of an ugly product makes life easier, as maximizing the logarithm of a function is equivalent to maximizing that function:

$$
\begin{align}
l(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1})
&= \log{L(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1})} \\
&= \sum_{i=1}^{n} \log{p(y^{(i)})} \\ &+ \sum_{i=1}^{n} \left( \sum_{j=1}^{d_i} \log{p(x_{j}^{(i)} \vert y^{(i)}; \phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1})} \right) \\
\end{align}
$$

We are modeling $ p(x^{(i)}|y^{(i)}) $ according to a multinomial distribution with parameters 
$ \phi_y $, $ \phi_{k \vert y = 0} $ and $ \phi_{k \vert y = 1} $, hence

$$
p(x_{j}^{(i)} \vert y^{(i)}; \phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) = 
$$

(Note that the function is not convex, unlike the Bernoulli case! Hence, we use Lagrange Multiplier)

