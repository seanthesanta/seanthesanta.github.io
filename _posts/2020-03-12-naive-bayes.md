---
title: "Naive Bayes: a brief introduction to generative models"
excerpt_separator: "<!--more-->"
categories:
	- Generative Models
tags:
author_profile: false
---

Yep, Naive Bayes is a generative model. But what is a generative model? To illustrate, let's say we want to distinguish a car and a bicycle. 

![A bicycle and a car](/assets/images/2020-03-12-naive-bayes/bikeAndCar.jpg)

As we already knew, logistic regression works with the model $p(y \vert x; \theta)$, which means it tries to find a set of parameters $\theta$ such that we can map each $x$ into a catagory $y$. In other word, it tries to learn the **classifier which separates a car and a bicycle, without having to know what is a car and what is a bicycle**. But now, let's say before distinguishing a car and a bicycle, we want to learn **how a bicycle or a car looks like** first. That's what a generative model do: it tries to find parameters $\theta$ 's such that they most fit in with the description of a car or a bicycle, i.e. $p(x \vert y; \theta)$.

Finally, we just need to compare our new input $x$ with the description that we learned to classify it!

For example, below is the visualization of a generative model with the elliptic contours representing a Gaussian distribution fitted to examples in corresponding class. These distributions are our "description" of the classes.

![Visualization of a generative model, from CS229](/assets/images/2020-03-12-naive-bayes/generativeVisual.png)

To convert from our "description" of a class $p(x \vert y; \theta)$ to our "classifier" $p(y \vert x; \theta)$, we use Bayes rule:

$$
p(y \vert x;\theta) = \frac{p(x \vert y; \theta)p(y)}{p(x)}
$$

With a general idea about a generative model, let us dive into Naive Bayes. Just like before, we consider a real life application, where we have to classify spam emails.

![SPAM!!!](/assets/images/2020-03-12-naive-bayes/spamMeat.jpeg)

Say, we have a dictionary $V$ of 35000 words of interest: a, aardvark, aardwolf, ..., zygumrgy. Each email is represented by a length-$d_i$ vector $x^{(i)}$, where $x_{j}^{(i)} = k$ if the $j$-th word in the email is the $k$-th word in the dictionary.

The problem arises when we try to model the whole email $x^{(i)}$ after a multinomial distribution. For example, if each email only has $100$ words, we still have $35000^{100}$ possible outcomes, hence we have to deal with a vector having $35000^{100}-1$ entries. This is very computationally expensive.

A solution to this problem is to assume that in each email, all words $x_j$'s are conditionally independent given $y$. **Basically, we are assuming that: if we are aware that an email is spam, knowing whether a word appears in that email does not influence our belief about the appearance of another word**. For example, in a spam advertising email, knowing the word "cheap" appears in that email does not affect our belief about whether the word "product" is present or not. Obviously, this is a *naive* assumption, since we know different words in an email are correlated, e.g. if a word "cheap" appears then it's very likely that "product" also appears (that email probably is advertising about some "cheap product"). Hence, the name *Naive* Bayes.

This *naive* assumption simplifies our task by reducing a high-dimensional distribution to just a product of low-dimensional ones:

$$
\begin{align}
p(x_1, x_2, ..., x_d \vert y; \theta)
&= p(x_1 \vert y; \theta) p(x_2 \vert y; \theta) ... p(x_d \vert y; \theta) \\
&= \prod_{j=1}^{d} p(x_j \vert y; \theta) \\
\end{align}
$$

We are going to model $ p(x_j \vert y) $ after a multinomial distrubution. Before that, we assume that $ p(x_j = k \vert y) $ is the same for every $j$, i.e. the distribution after which a word is generated is independent of the position of that word in the email. In other word, a word $j$ does not appear more frequently in a particular position and less at another place in an email. 

With the above assumption, we parameterize the distribution:

$$
\begin{align*}
\phi_y &= p(y=1)\\
\phi_{k \vert y=1} &= p(x_j = k \vert y = 1)\\
\phi_{k \vert y=0} &= p(x_j = k \vert y = 0)
\end{align*}
$$

Given a dataset of emails $ \{(x^{(i)}, y^{(i)}); i = 1,...,m\} $ with $ x^{(i)} \in \mathbb{R^{d_i}} $ where $ d_i $ is length of $ i $-th email, the likelihood of the dataset is

$$
\begin{align*}
L(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) 
&= \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}) \\ 
&= \prod_{i=1}^{m} p(x_1^{(i)}, x_2^{(i)}, ..., x_{d_i}^{(i)} \vert y; \phi_{k \vert y=0}, \phi_{k \vert y=1})p(y^{(i)}; \phi_y) \\
&= \prod_{i=1}^{m} \left( \prod_{j=1}^{d_i} p(x_{j}^{(i)} \vert y^{(i)}; \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) \right) p(y^{(i)}; \phi_y)
\end{align*}
$$

where the third equality is due to $(2)$.

Taking $ log $ of an ugly product makes life easier, as maximizing the logarithm of a function is equivalent to maximizing that function:

$$
\begin{align*}
l(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1})
&= \log{L(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1})} \\
&= \sum_{i=1}^{m} \log{p(y^{(i)}; \phi_y)} \\ &+ \sum_{i=1}^{m} \left( \sum_{j=1}^{d_i} \log{p(x_{j}^{(i)} \vert y^{(i)}; \phi_{k \vert y = 0}, \phi_{k \vert y = 1})} \right) \\
\end{align*}
$$

Let $v = \vert V \vert$ be the number of words in our dictionary. We are modeling $ p(x_j|y) $ according to a multinomial distribution with parameters 
$ \phi_y $, $ \phi_{k \vert y = 0} $ and $ \phi_{k \vert y = 1} $, hence

$$
\begin{equation}
p(x_{j}^{(i)} \vert y^{(i)}; \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) = \prod_{k=1}^{v} \left(\phi_{k \vert y=0}^{1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}} \phi_{k \vert y=1}^{1\{x_{j}^{(i)} = k \wedge y^{(i)} = 1 \}}\right)
\end{equation}
$$

We also have

$$
\begin{equation}
p(y^{(i)}; \phi_y) = \phi_y^{1\{y^{(i)} = 1\}}(1-\phi_y)^{1\{y^{(i)} = 0\}}
\end{equation}
$$

Plug $(3)$ and $(4)$ into the expression of log-likelihood $l(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1})$,

$$
\begin{align*}
l(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) 
&= \sum_{i=1}^{m}\left(1\{y^{(i)}=1\}\log\phi_y + 1\{y^{(i)}=0\}\log(1-\phi_y)\right)\\
&+ \sum_{i=1}^{m}\sum_{j=1}^{d_i}\sum_{k=1}^{v}[1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}\log\phi_{k \vert y=0}\\
&+ 1\{x_{j}^{(i)} = k \wedge y^{(i)} = 1 \}\log\phi_{k \vert y=1}]
\end{align*}
$$

Before optimizing $l(\phi_y, \phi_{k \vert y = 0}, \phi_{k \vert y = 1}) $, there's an important thing to note. The parameters $\phi_{k \vert y=0}$ and $\phi_{k \vert y=1}$ subject to $\sum_{k}^{v}\phi_{k \vert y=0} = 1$ and $\sum_{k}^{v}\phi_{k \vert y=1} = 1$ respectively. Therefore, we use Lagrange multipliers to optimize $l$.

Consider the corresponding Lagrangian $\hat{l}$, 

$$
\begin{equation*}
\hat{l} = l + \lambda_0(1-\sum_{k=1}^{v}\phi_{k \vert y=0}) + \lambda_1(1-\sum_{k=1}^{v}\phi_{k \vert y=1})
\end{equation*}
$$

with $\lambda_0, \lambda_1 \in \mathbb{R}$.

Setting the derivative w.r.t $\phi_y$ to $0$, we have:
$$
\begin{align}
\frac{\partial \hat{l}}{\partial \phi_y} = 0 
&\Leftrightarrow \frac{1}{\phi_y}\sum_{i=1}^{m}1\{y^{(i)} = 1\} = \frac{1}{1-\phi_y}\sum_{i=1}^{m}1\{y^{(i)}=0\}\\
&\Leftrightarrow \phi_y = \frac{\sum_{i=1}^{m}1\{y^{(i)}=1\}}{m}
\end{align}
$$

Setting the derivative w.r.t $\phi_{k \vert y=0}$ to $0$, we have

$$
\begin{align}
\frac{\partial \hat{l}}{\partial \phi_{k \vert y=0}} = 0 \Leftrightarrow \sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \} = \lambda_0\phi_{k \vert y=0}\\
\end{align}
$$

Setting the derivative w.r.t $\lambda_0$ to $0$, we have

$$
\begin{align}
\frac{\partial \hat{l}}{\partial \lambda_0} = 0 \Leftrightarrow 1 = \sum_{k=1}^{v}\phi_{k \vert y=0}\\
\end{align}
$$

From $(7)$ and $(8)$,

$$
\begin{equation}
\sum_{i=1}^{m}\sum_{j=1}^{d_i}\sum_{k=1}^{v}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \} = \lambda_0\sum_{k=1}^{v}\phi_{k \vert y=0} = \lambda_0
\end{equation}
$$

Plug $(9)$ back into $(7)$,

$$
\begin{align}
\phi_{k \vert y=0} 
&= \frac{\sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}}{\lambda_0}\\
&= \frac{\sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}}{\sum_{i=1}^{m}\sum_{j=1}^{d_i}\sum_{k=1}^{v}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}}\\
&= \frac{\sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}}{\sum_{i=1}^{m}1\{y^{(i)}=0\}d_i}
\end{align}
$$

Similarly, we have

$$
\begin{equation}
\phi_{k \vert y=1} = \frac{\sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 1 \}}{\sum_{i=1}^{m}1\{y^{(i)}=1\}d_i}
\end{equation}
$$

From $(6)$, $(12)$ and $(13)$, the optimal parameters are

$$
\begin{align*}
\phi_y &= \frac{\sum_{i=1}^{m}1\{y^{(i)}=1\}}{m}\\
\phi_{k \vert y=0} &= \frac{\sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 0 \}}{\sum_{i=1}^{m}1\{y^{(i)}=0\}d_i}\\
\phi_{k \vert y=1} &= \frac{\sum_{i=1}^{m}\sum_{j=1}^{d_i}1\{x_{j}^{(i)} = k \wedge y^{(i)} = 1 \}}{\sum_{i=1}^{m}1\{y^{(i)}=1\}d_i}\\
\end{align*}
$$

**REFERENCE**

[1] <http://cs229.stanford.edu/notes-spring2019/cs229-notes2.pdf>

[2] <https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf>
