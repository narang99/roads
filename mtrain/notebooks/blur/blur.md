# Thoughts dump

- The downsizing algorithms are reducing the "border size". This makes sense actually. But still.  
  - The actual photo seems "sharper" than the downsized photo. The borders are more pronounced, borders "pronouncing" gets lesser with resizes (note that right now I'm working by only resizing once, so the first resize itself is reducing the sharpness of the image compared to the original).  
- Gaussian blur smoothens images, it does not really blur them as in "distant versions" of the image


# Conclusion exp001-plastic-bag
- Counting pixels isn’t very objective right now because I can’t discern what a border is after a single point.  
Although I can see it at a higher level, the border itself gets hazy when you look down on it. When you look closely, it’s difficult to tell whether a pixel belongs to the border.  
- The method was to take photos of the same object at different distances. We are trying to translate the object from a closer distance to a farther distance. The difference between the resized closer object and the original farther object is that the resized version is less sharp than the original farther version.   
- The number of pixels that compose the border is less, I'm guessing. I have to test it. A good, faithful algorithm for downscaling might be a better resizing algorithm instead of using different blur types.  

Next step: Check the area difference of what composes a border in resized version compared to the original version

# exp002-plastic-bag

I'm trying to find a resize algorithm which keeps the intensities good.  
For very far away images, I'm assuming that the dominant intensity of the object is caught. I might be wrong here.  

For objects which still have visible details, I believe that we do still see the high contrast. the whole spectrum of that object is caught.  
Linear interpolation looks very artificial to me. Nearest neighbors infact looks better when the scaling is small (1.5x ish).  
When the scaling ratio is high, we see extremely smooth images coming up in linear interpolation, unreal smooth images.  



NOTE: I'm concerned if I'm going down a rabbit hole which is already researched and considers that what I'm doing is not correct.  

Given an image A, and scale ratios `r[x]` and `r[y]`, A square of `r[x] * r[y]` size needs to be converged to a single pixel.  


# Interpolation techniques

Linear interpolation is simple when expanding the image, it fills pixels between two pixels by following a line.  
Cubic interpolation uses 4 points in a line, and fits a 3rd degree polynomial to it (It's not very hard to find the solutions of this polynomial). This ends up being a smoother interpretation of the function for interpolation (simply because it has more movement and is more complex).  

I need to see how it happens for downscaling though.  
GPT keeps talking about aliasing while downsampling. 
Theres a theorem: Given a sample of frequency f, you need to sample 2f points to come up with that frequency.  
Here frequency is the change of intensities (I'm thinking that this is related to different regions of interest in the image). Naturally if you sample less points that the actual number of intensity changes, you will lose the number of intensity changes. GPT says bluring and interpolation helps here, but I'm not sure what it means by that.  


INTER_NEAREST is really good I feel. Aliasing needs to be understood and figured out.  