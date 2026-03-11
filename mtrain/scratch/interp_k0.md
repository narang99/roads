# Edge kernels

Spread: the pixel length activations exist before flattening to empty space again. Counted by running a kernel on a step edge, while aligning the kernel at the place where it detects edges. A kernel can detect edges at multiple alignments. We find spread for all alignments.  

The other part I need to think about is magnitude, or the scaling factor, which I won't discuss right now. Might do it later.  

Sum: simple sum of a kernel to see whether the edge kernel is increasing the brightness? Generally edge kernels would have a sum ~ 0.  

# Simplenet: Layer 0, kernel 0

Conv2D layer:
```
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```


All kernels look quite similar. They serve the same function, edge detection.  
Since the kernel has a stride of 2, a normal sobel like kernel would only be able to detect half of the edges, where it aligns with the edge (stride 2 can either align sobel at edges at odd places, or even places).   

These are 7x7 kernel, with 2 alignment points. The first alignment point results in very strong activations for a step edge. These align at step edges present at even locations in the image (for the given padding).   

# Horizonal edges

## Kernel stats
```
Raw Kernel sums:
	 0 0.15076977
	 1 0.021285338
	 2 0.03596267
total 0.20801778
```

```
Absolute Kernel sums:
	 0 6.808271
	 1 8.928421
	 2 4.9532084
total 20.689901
```

## Strong alignment stats
```
Raw Per channel sums:
	 0 -23.47058
	 1 -7.2561913
	 2 15.275373
total -15.451397
```

```
Abs Per channel sums:
	 0 469.9002
	 1 531.71436
	 2 194.79636
total 1196.4109
```

```
Output raw sum -15.451714

Output abs sum 1183.4376
```

```
Green maxes:
    0 4.6778126
    1 7.340943 
    2 2.3656154 
```

```
Red mins:
    0 -4.0712733
    1 -6.4337554
    2 -1.3005104
```

## Weak alignment stats
```
Raw Per channel sums:
	 0 -68.8347
	 1 -71.88545
	 2 -49.848434
total -190.56859
```

```
Abs Per channel sums:
	 0 379.75546
	 1 331.65866
	 2 219.20436
total 930.61847
```

```
Output raw sum -190.56891

Output abs sum 922.7656
```

```
Green maxes:
    0 1.8185747
    1 1.3341014
    2 1.7368298
```

```
Red mins:
    0 -3.1071162
    1 -3.8291485
    2 -2.886985 
```

## Compare

- kernel c[0] and c[1] give strongest activations in strong alignment at green maxes and red mins
- kernel c[2] has similar activation strength in both strong and weak alignments
- Green maxes are similar for all kernels in weak alignments
- Red mins are also similar

- If kernel c[0] or c[1]'s strong alignment coincides with any of the weak alignments, they would dominate
- greens of weak alignment are weaker than its reds, making the total chan sum net red
    - not the case for strong, both bands are close in abs value, cut each other out

## Test tweaking the kernel
Kernel c[0] and c[1] complement each other. They are the stronger ones in the channel and their sums always add up to find edges if the input edges are aligned.  
The test simply marked `c[1] = -c[1]`, ie invert the first channel.  

This changed the model's prediction changed from trash to other when we did this. The picture is that of a slipper. The edge kernel was activating at the large horizontal part of the slipper originally. When inverted, the activations cancelled and the object was marked other.  

The model was relying on the slipper's horizontal patch for its work it seems.  

`c[2]` is very weak compared to the other two. to confirm it, I tweaked it by:
- zeroing it out
- inverting it

in both cases, its not giving me any changes in the test image set (it is possible i might need to create a larger set though. I'm not doing it right now so that it does not get overwhelming).  

## c[0] and c[1]

Edges are detected in two manners. 

Edge 1:
- Spread: 2
- Color alignment: negative (red when green -> red, green when red -> green)
- Activations, quite high abs values

Edge 2:
- Spread: 3
- Color alignment: positive (green when green -> red, red when red -> green)
- Activations: a lot lower than Edge 1

This edge has a black band as the middle portion. 

Depending on padding, 
- edge 1 would always come when the edge is at an even number
- edge 2 would always come when the edge is at an odd number
Or vice versa (edge 1 on odd, edge2 on even)
The fact that these kernels have two types of edges helps it detects edges at every pixel length even though the kernel itself is used with a stride of 2 in this layer.  

It's useful to see the notebook to see how these edges come up:
### Edge 1

- alignment with horizontal edge in input at row 5 or row 2
  - either one of the opposite greens changes signs, or both reds change signs. In this case, they starting ganging up.
  - The activation will basically be the pointwise multiplications of the mid red portions for Edge 1 type

### Edge 2
- alignment at row 6 or row 1
  - this basically ends up with row 1 and 6's balance going out, and one of them wins. the magnitude is smaller though since the outer rows are weaker than the inner rows



# Vertical edges

```
Raw Per channel sums:
	 0 13.75264
	 1 -3.6905303
	 2 2.0984712
total 12.16058

Abs Per channel sums:
	 0 263.07507
	 1 194.16264
	 2 98.31565
total 555.5534

Output raw sum 12.160164

Output abs sum 548.4108
```

The raw channel sums are larger than the horizontal case because there is no strong red band. The abs sums are smaller, supporting this hypothesis.  


The edges we are analysing are of spread 3
```

coordinate: 10,11
green maxes
	 0 0.33906746
	 1 0.23882532
	 2 0.21893445

red mins
	 0 -0.31931594
	 1 -0.059539974
	 2 -0.18476412

coordinate: 10,12
green maxes
	 0 0.33906746
	 1 0.23882532
	 2 0.21893445

red mins
	 0 -0.31931594
	 1 -0.059539974
	 2 -0.18476412

coordinate: 10,13
green maxes
	 0 0.33906746
	 1 0.23882532
	 2 0.21893445

red mins
	 0 -0.31931594
	 1 -0.059539974
	 2 -0.18476412
```

The maxes and mins are weaker than the weak horizontal edge detection.   

`c[0]` does seem to be doing a weak detection of vertical edges. The kernel slice just has a total sum = 0.13. When it is on green side, the whole thing has a slight green hue. when it is on the red side, the whole thing has a slight red hue. when the edge is at the mirror of this kernel (column 4) (column 4 seems to be the symmetry place for this one), the whole thing becomes dark.  

For `c[2]`, the reds are stronger. when they are on a red side, they end up giving a good green hue, 


The vertical edge functions seem to be quite different from the horizontal edge functions.  

- for `c[0]`, when we go from green to red, the value of each column starts at the top and consistently goes down to the last stable value. It seems to "smooth" out the edge.  
- `c[1]` first goes up to a maximum (the maximum is one column on right of edge [in the red side]). And then sharply falls to the stable value on the right
- `c[2]` has the same behavior as `c[1]`

I honestly feel right now that `c[0]` is actually a blurring kernel when it goes to the right. I'll have to verify this