# Basics

Linear algebra seems to be a study of frames of reference and how they change. A single frame of reference in linear algebra is basically like a number line. In my number line, B represents the length of the unit along the line, and that is the length of basis vector along its direction.  
A vector along a basis is basically a constant multiplied by that basis vector. The multiplication part is how we emphasize the direction of the basis. I think this is why the parallelism between dimension grades is preserved. Because it's a flat surface and all distances are equal initially, and the grid is parallel at the start, if you double the distance at every cell in the grid, it remains parallel and equidistant. This seems to be a core property of multiplication. I need to later analyze it in my number line work.
Given two vectors, we add them by summing their projections along the basis vectors. This is similar to how composition is additive in the real world, like on a number line. This is the classic example of how you would model displacement.

A single vector is simply specified like so: `[a b]`, where `a` is the projection along the first basis vector, `b` is the projection along the second basis vector.

Addition is defined by simply adding individual components. No multiplication is defined between vectors. Scalar multiplication is defined, which simply stretches/squeezes the vector along one dimension.

# Translating between frames

The same vector in its absolute position in space can be defined by any frame of reference I choose. It uses i and j, which are perpendicular, but you could define any random frame of reference with two non‑parallel lines. You need those two non‑parallel lines for a two‑dimensional grid. Now how do we translate with our representations between these frames of reference?

The construction here proves how a basis change can be done
![Basis change construction](./lin-alg-board.jpeg)

Essentially, if some vector `V` is `[a, b], {u,w}`, where `u = [c, d], {i,j}` and `w = [e, f], {i,j}`, then:

- You can do a simple substitution `au = a([e d], {i,j})`, `bw = b([e, f], {i,j})`, if you look at the constrution, you will see that the `{i,j}` and `{u}` form similar triangles with `V = [a]{u}`. So a simple multiplication works.
- Now you need to think in terms of displacement. You simply walk to `au` in terms of `i,j`, then walk `bw` in terms of `i,j`.
- Look at the construction. You will see that the values are added individually in each axis.

And so, the transformation essentially gives `[ac + be]` as the `i` coordinate for `V`. `[ad + bf]`. I feel the notation in 3blue1brown is wrong. You can't do a weird scalar multiplication there.
It makes more sense to do this construction. And define matrix multiplication as this (instead of playing around with basis) randomly.
It does not change the fact that the matrix is a representation of how basis vectors are represented in the our frame of reference.

# Transforming in a single frame

until now, we saw a definition of matrix multiplication in the form of representing something in our basis, given that vector in another basis. The vector does not change absolute position or direction at all, only it's representation does.

What happens when we multiply this matrix to something represented in our own space? The intuitive way is to say:

- Matrix multiplied to our basis gives the new basis
- They are "transformed linearly" (which basically means there is an add + multiply combination)
- Since ratios are preserved, when doing this multiply/add trick a vector represented in our space, it transforms the vector in a "similar" way as seen in our basis

So now, if the matrix rotates our grid, the vector is also equally "rotated". This is weird to me.
The intuition to go from basis representation to transformation is not very clear.

Take the example of rotation. A grid which is 90 degree rotated from us, has a basis vector `u = [0, 1]{i,j}`, `v = [-1,0]{i,j}` in our space.
The matrix `[[0,-1], [1,0]]` represents this translation. If we look at the basis vector `u = [1,0]{u,w}` in our space, it comes up to be `[1, 0]{i,j}`. This is how we represent it in our space.
The problem here is, we want to see a vector in our space, with this transform applied, and think of those coordinates as coordinates IN OUR SPACE. This changes the vector.

I've got it now:

- A matrix multiplication simply moves the vector to a new place, where that new vector has the same coordinates as the old vector in the new basis system. Simple.

`[3, 2], {i,j}` moves `3` in the `i` direction and `2` in the `j` direction.  
We now instead move `3` in the `u` direction and `2` in the `w` direction, and get `[3,2], {u,w}` and get a new vector in space. This vector can also be written in our notation using change of basis matrix. So we say that multiplying change of basis matrix moves the vector that new vector.

# Composition

Let's see if linear transformation is commutative/associate

For some reason, when I do matrix composition geometrically `AB`, I get the results `BA` lol, and vice-versa. I'm somehow doing them in reverse hehehehe ;). This is mostly because of my change of basis matrix definition.

## Rotation and shear

We have two matrices—one for rotation and one for shear. We want to see how the grid looks geometrically when we apply a shear followed by a rotation, and vice versa.
If we apply the matrix multiplications algebraically to the i basis vector, something weird happens. So let's say you apply the rotation matrix first. The first column of the rotation matrix shows where the basis vector **î** ends up. The second column shows where **ĵ** ends up. If we only care about the final position of **î**, we don’t need to consider how **ĵ** looks after the intermediate step.

It’s a bit odd, but in the intermediate transformation the final position of **î** is independent of where **ĵ** was after the first transformation. That intermediate state of **ĵ** has no effect on **î**.

When we apply the rotation matrix to our basis vector \(\hat{i}\), \(\hat{i}\) ends up where \(\hat{j}\) is in the current frame. After that, applying shear does not act on the new basis vector in the new basis; it acts on the transformed \(\hat{i}\) in the original \(\mathbb{R}\) basis frame.

The effect should be viewed as if it were applied directly to \(\hat{i}\) and \(\hat{j}\), not to the intermediate frame resulting from the first matrix multiplication.

Let's see how I was doing this transformation. I first applied the rotation.
Then `i` becomes `j` (`[0,1]`). The Shear defines that `j` goes to `[1,1]`. This makes `i` move to `[1,1]`. This is quite different from what I did in the beginning.

I assumed, that the Shear would work in the new basis. After the rotation, we get a new basis `u = [0, 1]`, `w = [-1, 0]`

The problem is that the intermediatge vector after multiplication is not `[1,0]`. Even if we think of the basis changing in the new basis space, the new vector being moved is NOT [0,1] (the basis in the new frame). Its another vector in the new frame lol, which is then transformed to the final space.

# Determinant

Although we are doing a geometric interpretation of linear algebra, for now I think it is important to note that linear algebra did not arise from geometry alone. The concepts have been around for a long time, originating from people trying to solve systems of equations. The determinant is not defined simply as the scalar by which the area between basis vectors is scaled; it has a more complex definition that I don't fully understand.

From what I gather, people were working with linear equations and properties related to determinant calculation. The first simple observation is that if the determinant is zero, then the linear system is unsolvable.

Later, the scope of the determinant was expanded, and someone showed how it fits nicely into a broader framework. The geometric interpretation is actually a consequence of that development.
