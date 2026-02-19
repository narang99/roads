References:
- https://www.youtube.com/watch?v=mTw3o8-xMIo
- Book: Hartley & Zisserman – Multiple View Geometry (Ch 1)

Projective Geometry is the study of how planar surfaces map on one another. An example, a single plane of 3D scene, on a 2D mirror. All the definitions seem quite complicated (https://en.wikipedia.org/wiki/Projective_geometry). I started studying it to understand how a 3D scene is mapped on a 2D plane in a pinhole camera.

These people are trying to use concepts of euclidean geometry (at least that's what everyone is saying).   

# Points of infinity

This is the first thing everyone is talking about. When you map parallel lines in a 3D scene to a 2D plane, they seem to intersect at vanishing points (the horizon is called the infinity line). This mythical point will come up somewhere in the projected 2D surface concretely. Euclidean geometry does not traditionally deal with infinity points. Projective geometry however, adds these as standard points which "exist".  
A lot of stuff is defined as "projective transformation". Wikipedia points to simply an article on homography.

A projective transformation is hard to figure out at first. It does not preserve angles, lengths, and shapes. The first thing we notice is that it preserves straight lines (not their lengths or angles, but the fact that they are straight). A straight line in space 1 would be straight in space 2 after the projective transformation.  

Note that even ratios are not maintained in projective transformations. Something like 'ratio of the ratio of distances' is preserved.
This is called the cross ratio. Quite interesting, god knows who noticed this.

Although everyone is talking about projective transformations from plane to plane, I'm not sure if we are giving a special treatment to the mechanics of the actual 3d scene (which follows euclidean geometry).  
What is interesting about this study I feel is simply the existence of a vanishing point. That parallel lines meet (for humans too). The world we see is not the true world in this sense.
