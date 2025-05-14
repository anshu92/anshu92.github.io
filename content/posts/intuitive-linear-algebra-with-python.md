---
date: "2025-05-13"
draft: false
title: "🐍 Python Guide to Linear Algebra for Machine Learning for Engineers"
description: "A code-first, step-by-step exploration of linear algebra using Python, and its applications in machine learning."
categories: ["Machine Learning", "Math", "Linear Algebra", "Python"]
tags: ["math", "linear algebra", "vectors", "python", "code-first"]
math: true
---
{{< katex >}}
This series is designed to forge a deep, intuitive understanding of the mathematics that power the code I write every day as an ML engineer. Just as a traveler in Montreal might scrape by on a handful of French phrases but needs true fluency to think and live there, we often learn machine-learning theory and then struggle to translate it into code. Likewise, many engineers can train models and assemble practical toolkits without ever unpacking the mechanics beneath the surface. Here, we'll sidestep heavy libraries like NumPy—at least at first—to examine each calculation by hand and demystify exactly how the math drives every line of code.

## Chapter 1: Dot Product

Let's start by implementing the dot product, the workhorse computation of machine learning. 

### Calculating the Dot Product

Given two vectors, say \\(\vec{a} = [a_1, a_2, \dots, a_n]\\) and \\(\vec{b} = [b_1, b_2, \dots, b_n]\\), their dot product is found by multiplying corresponding elements and summing the results.

```python
import math

# Dot product of two vectors
# [a1, a2, a3], [b1, b2, b3] -> a1*b1 + a2*b2 + a3*b3
def dot_product(v1, v2):
    # Ensure vectors are of the same length
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(x * y for x, y in zip(v1, v2)) # all hail zip!!

# Example vectors
a = [1, 3, -5]
b = [4, -2, -1]

# Calculate and print the dot product
ab_dot_product = dot_product(a, b)
print(f"Vector a: {a}")
print(f"Vector b: {b}")
print(f"Dot product (a · b): {ab_dot_product}")
````

This will output:

```
Vector a: [1, 3, -5]
Vector b: [4, -2, -1]
Dot product (a · b): 3
```

**What's happening here?**

The formula we've just implemented is:

$$
\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n
$$For our example vectors \\(\\vec{a} = [1, 3, -5]\\) and \\(\\vec{b} = [4, -2, -1]\\):

$$\\vec{a} \\cdot \\vec{b} = (1)(4) + (3)(-2) + (-5)(-1) = 4 - 6 + 5 = 3
$$The result, `3`, is a single number (a scalar), which is why the dot product is often called the **scalar product**.

### The Geometric Intuition: Angles and Magnitudes

The real magic of the dot product comes from its geometric interpretation:

$$
\vec{a} \cdot \vec{b} = \|\vec{a}\| \|\vec{b}\| \cos(\theta)
$$Where:

- \\(|\\vec{a}|\\) is the magnitude (or length) of vector \\(\\vec{a}\\).
- \\(|\\vec{b}|\\) is the magnitude of vector \\(\\vec{b}\\).
- \\(\\theta\\) is the angle between vectors \\(\\vec{a}\\) and \\(\\vec{b}\\).

Let's add functions to calculate the magnitude of a vector and the angle between two vectors.

```python
# Magnitude of a vector
def magnitude(v):
return math.sqrt(sum(x**2 for x in v))

# Angle between vectors in degrees
def angle_between(v1, v2):
dp = dot_product(v1, v2)
mag1 = magnitude(v1)
mag2 = magnitude(v2)

# Prevent division by zero if one of the vectors is a zero vector
if mag1 == 0 or mag2 == 0:
return 0 # Or raise an error, or handle as appropriate

# Clamp the cos_theta value to the range [-1, 1] to avoid math domain errors with acos
cos_theta_val = dp / (mag1 * mag2)
cos_theta = max(-1.0, min(1.0, cos_theta_val))

theta_rad = math.acos(cos_theta)
return round(math.degrees(theta_rad), 2)

# Calculate magnitudes
mag_a = magnitude(a)
mag_b = magnitude(b)

# Calculate angle
angle_ab = angle_between(a, b)

print(f"Magnitude of a (||a||): {round(mag_a, 2)}")
print(f"Magnitude of b (||b||): {round(mag_b, 2)}")
print(f"Angle between a and b (θ): {angle_ab} degrees")
```

This will output:

```
Magnitude of a (||a||): 5.92
Magnitude of b (||b||): 4.58
Angle between a and b (θ): 83.61 degrees
```

**Interpreting the Dot Product's Sign:**

The formula \\(\\vec{a} \\cdot \\vec{b} = |\\vec{a}| |\\vec{b}| \\cos(\\theta)\\) tells us a lot about the vectors' orientation:

- If \\(\\vec{a} \\cdot \\vec{b} > 0\\): \\(\\cos(\\theta) > 0\\), meaning \\(0^\\circ \\leq \\theta < 90^\\circ\\). The vectors point in the **same general direction**.
- If \\(\\vec{a} \\cdot \\vec{b} = 0\\): \\(\\cos(\\theta) = 0\\), meaning \\(\\theta = 90^\\circ\\). The vectors are **perpendicular** (orthogonal). This is a crucial concept in many areas of math and computer science.
- If \\(\\vec{a} \\cdot \\vec{b} < 0\\): \\(\\cos(\\theta) < 0\\), meaning \\(90^\\circ < \\theta \\leq 180^\\circ\\). The vectors point in **opposite general directions**.

Our result of `3` (positive) and an angle of `83.61` degrees aligns with this: the vectors are generally pointing in a similar direction.

### Applications in Machine Learning and Beyond

The dot product isn't just an abstract mathematical operation; it's incredibly useful:

1.  **Cosine Similarity**: The term \\(\\cos(\\theta)\\) can be isolated: \\(\\cos(\\theta) = \\frac{\\vec{a} \\cdot \\vec{b}}{|\\vec{a}| |\\vec{b}|}\\). This is the cosine similarity, a metric ranging from -1 to 1 that measures how similar the *direction* of two vectors is.

* **Search Engines & Recommendation Systems**: Used to compare query vectors with document vectors or user preference vectors with item vectors. High cosine similarity means high relevance.
* **Natural Language Processing (NLP)**: Word embeddings (like Word2Vec, GloVe, or those from Transformers) represent words as vectors. Cosine similarity between these vectors can indicate semantic similarity between words.
* **Large Language Models (LLMs)**: Attention mechanisms, a core component of Transformers (which power LLMs like GPT), heavily rely on dot products to determine how much "attention" one part of a sequence should pay to another. Essentially, it's calculating similarities between query, key, and value vectors.

2.  **Projections**: The dot product helps in projecting one vector onto another. This tells us how much of one vector lies in the direction of another. Imagine shining a light perpendicularly onto vector \\(\\vec{b}\\); the length of the shadow of \\(\\vec{a}\\) on \\(\\vec{b}\\) is related to the dot product.

3.  **Geometric Transformations**: As we'll see with matrices, dot products are fundamental to rotating, scaling, and shearing vectors in 2D/3D graphics.

4.  **Physics - Work Done**: A classic example. If a force \\(\\vec{F}\\) acts on an object causing a displacement \\(\\vec{d}\\), the work done (\\(W\\)) is:

$$
W = \vec{F} \cdot \vec{d}
$$Only the component of the force that is *in the direction of movement* contributes to the work. If the force is perpendicular to the displacement, the dot product is zero, and no work is done by that force in that direction. (If your attention vector while reading this is perpendicular to the content, your learning "work done" might be zero\! 📐😉)

### Extending to Matrix-Vector Products

Now, let's see how the dot product extends to multiplying a matrix by a vector.

Consider a matrix \\(A\\) and a vector \\(\\vec{x}\\):

```
A = [
  [a₁₁, a₁₂, ..., a₁ₙ],  // row₁
  [a₂₁, a₂₂, ..., a₂ₙ],  // row₂
  ...
  [aₘ₁, aₘ₂, ..., aₘₙ]   // rowₘ
]

x = [x₁, x₂, ..., xₙ]ᵀ // Transpose for column vector notation
```

The product \\(A \\cdot \\vec{x}\\) is a new vector \\(\\vec{y}\\), where each element of \\(\\vec{y}\\) is the dot product of a row from \\(A\\) with the vector \\(\\vec{x}\\).

```python
# Matrix-vector dot product
def matrix_vector_product(matrix, vector):
    # Ensure the number of columns in the matrix matches the length of the vector
    if not matrix or (len(matrix[0]) != len(vector)):
        raise ValueError("Number of columns in matrix must match length of vector")
    return [dot_product(row, vector) for row in matrix]

# Example Matrix and Vector
A = [
    [2, 1, 0],
    [-1, 3, 2],
    [0, 0, 1]
]

x_vec = [4, -2, -1] # Using our previous vector b as x for this example

# Calculate and print the matrix-vector product
y_vec = matrix_vector_product(A, x_vec)

print(f"\nMatrix A:\n{A[0]}\n{A[1]}\n{A[2]}")
print(f"Vector x: {x_vec}")
print(f"Matrix-vector product (A · x): {y_vec}")
```

This will output:

```
Matrix A:
[2, 1, 0]
[-1, 3, 2]
[0, 0, 1]
Vector x: [4, -2, -1]
Matrix-vector product (A · x): [6, -12, -1]
```

**What does this mean?**

The resulting vector \\(\\vec{y}\\) is:

$$
\vec{y} = \begin{bmatrix} \text{row}_1 \cdot \vec{x} \\ \text{row}_2 \cdot \vec{x} \\ \vdots \\ \text{row}_m \cdot \vec{x} \end{bmatrix}
$$For our example:
\\(y_1 = \\text{row}_1 \\cdot \\vec{x} = [2, 1, 0] \\cdot [4, -2, -1] = (2)(4) + (1)(-2) + (0)(-1) = 8 - 2 + 0 = 6\\)
\\(y_2 = \\text{row}_2 \\cdot \\vec{x} = [-1, 3, 2] \\cdot [4, -2, -1] = (-1)(4) + (3)(-2) + (2)(-1) = -4 - 6 - 2 = -12\\)
\\(y_3 = \\text{row}_3 \\cdot \\vec{x} = [0, 0, 1] \\cdot [4, -2, -1] = (0)(4) + (0)(-2) + (1)(-1) = 0 + 0 - 1 = -1\\)

So, \\(\\vec{y} = [6, -12, -1]\\).

**Applications of Matrix-Vector Products:**

- **Weighted Sums in Neural Networks**: This is *exactly* how inputs are combined in a neuron. If \\(\\vec{x}\\) is a vector of inputs and a row of matrix \\(A\\) contains the weights for those inputs, their dot product is the weighted sum, which then goes into an activation function. The entire matrix \\(A\\) can represent the weights of a layer.
- **Systems of Linear Equations**: A system of linear equations can be compactly written as \\(A\\vec{x} = \\vec{b}\\), where we solve for \\(\\vec{x}\\).
- **Linear Transformations**: In geometry, multiplying a vector by a matrix can represent a linear transformation like rotation, scaling, or shearing of that vector. Each row of the matrix contributes to transforming the input vector into the output vector. The resulting vector \\(\\vec{y}\\) tells you where the vector \\(\\vec{x}\\) lands after being transformed by \\(A\\).

**Phew! That's a lot of work to understand something basic, but upon this stone we shall build our non-religious gathering place to preach the gospel of ML!**

## TO BE CONTINUED