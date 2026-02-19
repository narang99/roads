# Essence of Linear Algebra
https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab


# Matrix multiplication 
https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3

Matrices are linear transformations. They change how the grid looks. There are constraints on how the grid can look. The grid needs to be evenly spaced (as before, although the space was 1). And the grid should only move around the origin.  
This essentially means we change the basis vectors. `i` vector is changed to some other vector (like `1i + 2j`) in the older space. `j` moves to say `2i + j` space

In this case, the original vector in `i,j` basis also moves to a new position. The relative ratio to the new `i` and the new `j`, along with the direction, remain constant. Say the old vector was `i+3j`, in this case, the new vector becomes `(1i+2j) + 3(2i+j)` (replace `i,j` with new definitions). This becomes `7i + 5j`. The function which moves the basis to new basis can be simply described by 4 numbers, `1,2,2,1`. These are stacked as a matrix. Matrix is therefore a linear transformation function. The stack can be `[[1,2], [2,1]]`
A good example is rotation, let's say we rotate 90 clockwise. `i` becomes `-j` and `j` becomes `i`, `[[0 1], [-1, 0]]`. Its very intuitive now, how rotation can be done.

This also generalizes to higher dimensions.

# determinant, rank, inverse, column space, null space

When we apply a linear transformation, we want to know how the area of a one by one square changes in the older space to the newer space. The factor by which the area changes is called the determinant. Since this is a linear transformation where the spacing between grid lines is preserved and the grid lines remain parallel, any arbitrary area also changes by this factor.  

I did some calculation by finding the square and how it looks in the new dimension after transformation. It ends up becoming a parallelogram. The area ends up being surprisingly easy. For a matrix [[a, b], [c, d]], it is `ad - bc`.  

If the determinant is zero, the area scales to zero. In a two‑dimensional coordinate space, after matrix multiplication the area becomes zero, so everything collapses into a line and we lose a dimension.   
If we increase our dimensions to three, a matrix could drop everything into the second dimension by converting all vectors to a 2‑D space. In that case the determinant is zero. The matrix might also convert everything into a single line instead of a plane. All the vectors would be mapped to a single line in a 1‑D space, which also gives a determinant of zero.  
A matrix that collapses something into k dimensions is called a matrix of rank k. 
If the rank is high  as it can be (a 3D matrix creates a space of 3D), its called a full rank.  

---
Linear algebra is useful to solve linear systems of equations.  
You can write linear systems of equations by placing the coefficients in a matrix and the variables in a column vector. The operation is simply matrix multiplication. (Ax = B).  
In this case, to find X you need the inverse of matrix A. You can think of the inverse as the reverse transformation of the original transformation that A produces.
One line of finding this out can be by finding the older basis vectors in terms of the new basis vectors. Then you simply have your inverse matrix.

If the determinant is zero, the inverse does not exist (its impossible to have a linear transformation which converts a 2D space to a FULL 3D space [where every coordinate is covered])

---

When a transformation squishes dimensions (reduces dimensions), a number of initial input vectors would result in becoming the null vector (`[0,0]`). This is called the null space. I think this terminology is used because any linear system can be reformulated to solve for the null space, with everything on the right-hand side set to zero. You do this by adding an extra dimension for the constant coefficients. The null space is also called the kernel.

A set of all outputs after the transformation using a matrix is called the column space of the matrix. This is reasonably easy to understand. We have the basis vectors that form the matrix, and the span of those basis vectors is basically the column space.

Non-square matrices would change the dimension of the output space from the input space. A matrix like `[1,2,3], [3,4,5]` has two 3 basis vectors defined in two dimensions only. This means that it would map a 2D vector into a plane on a 3D space.

# Dot product

Given vectors `[[1], [2]], [[3],[4]]`, the dot product is the sum of corresponding entries in each vector `1*3 + 2*4`. Geometrically, it is basically `ABcos theta` (the projection of B on A, multiplied by magnitude of A).  
I'm honestly not sure about the significance here. I was going to normalize the projection using B's length instead of multiplying it by B's length. But many concepts in physics resemble dot products, which is why this has been standardized.  
In the playlist we see another way of thinking about dot products. The vector `[[1],[2]]` can be thought of as a matrix `[1,2]` (a 1x2 matrix). A linear transformation now. If you do the matrix multiplication, it gives the answer equal to the dot product. This has geometric meaning.  

The matrix `[1,2]` collapses a 2D space into a 1D single number line (its a Rank 1 transformation).  Turns out, the number line on which this collapses the space is simply the vector `[[1], [2]]` (you can derive this by finding how the older basis vectors are projected on this line.). So the dot product really is also a part of the linear transformation which collapses vectors in the 2D space to the number line `[1,2]` here.

Here he is again, not giving the motivation for this specific formula—why you would need to find the projection of one vector onto another, which does seem useful. But why would you want to multiply the magnitudes together?  

# Cross product

The cross product is simply the area of the parallelogram formed by two vectors. When I tried to find the determinant of a matrix, I went through a derivation that involved changing the unit square spanned by **i** and **j** to the basis vectors of the transformed space.

I found the area of the parallelogram to compute the determinant. Now we can also say that the cross product is the determinant of a matrix created from those two vectors.

Oh well, no. Cross product gives a vector, which is perpendicular to the plane spanned by the two vectors which are being cross producted. Its magnitude is the area of the parallelogram made by the initial vectors.  
Although this is the geometric meaning of the cross product, I still don't understand its motivation. It feels like just explaining how to compute the cross product and nothing more.

A lot of linear algebra concepts seem to come from physics needs. For example, the cross product indicates the axis of rotation. I don't know how the magnitude corresponds to the rotation.
