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