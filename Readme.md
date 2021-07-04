# Dive into Deep Learning book repo

## Installation

https://d2l.ai/chapter_installation/index.html#subsec-gpu

![](installation.png)

-----

# Ch1 Introduction 

### Types of machine learning

1. Regression

2. Classification

3. Tagging - tagging in medical journels

4. Recommendation systems

5. Search - order matters

6. Sequence learning

7. Unsupervised learning - distributing shift - test and train data sufficiently differ.

### Innovations along the way

1. Dropout

2. arrentionmechanism

3. multi stage designs

4. generative adverserial network

5. parallel computing

## 1.9. Exercises

- Which parts of code that you are currently writing could be “learned”, i.e., improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?

- I want to make a summary writing tool

- Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using deep learning.

- creating speech fakes

- Viewing the development of AI as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal? What is the fundamental difference?

- more coal more steam, here the quality of data is important

Where else can you apply the end-to-end training approach, such as in Fig. 1.1.2, physics, engineering, and econometrics?

- philosphical experiments

----

# Ch2 Preliminaries

## 2.1 Mathematical operation

- In mathematical notation, we would denote such a unary scalar operator (taking one input) by the
signature f : R → R. This just means that the function is mapping from any real number (R) onto
another. Likewise, we denote a binary scalar operator (taking two real inputs, and yielding one
output) by the signature f : R, R → R.

- Element wise operation :
Given any two vectors u and v of the same shape, and a binary
operator f, we can produce a vector c = F(u, v) by setting ci ← f(ui, vi) for all i, where ci, ui, and vi are the ith elements of vectors c,u, and v. Here, we produced the vector-valued F : Rd,d → Rd
by lifting the scalar function to an elementwise vector operation.

### Braodcasting mechanism

-  Used by pytorch when dealing with unequal tensor

### saving memory

- Every time we allocate a tensor the memory location is initialised
- So we have the cnoceptof inplace vector

### Ocnverting to other python objects

- converting to a numpy tensor

- converting to a scalar using torch,item.

### Exercises
1. Run the code in this section. Change the conditional statement X == Y in this section to X <
Y or X > Y, and then see what kind of tensor you can get.

- done

2. Replace the two tensors that operate by element in the broadcasting mechanism with other
shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

![](broadcasting.png)

page 67








