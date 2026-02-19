I need to study how I can shrink higher resolution photos into lower resolution in such a way that we get similar objects as part of a bigger scene. This would require learning image formation. And the parameters of how light works with cameras.  

Reference:
- https://www.youtube.com/playlist?list=PL2zRqk16wsdr9X5rgF-d0pkzPdkHZ4KiT

# Noise in images

Types:

- Photon shot noise
    - The amount of photons that arrive at the sensor for a specific point in scene are random. This makes your photo different at different points in space and time. It depends on the brightness of the point (not sure what I would do with this noise, its not very intuitive also)
    - Mostly, this changes brightness from point to point even though the overall sunshine is the same
    - I'm confused on how I can use this. It follows a poisson distribution
- Read noise
    - The noise introduced by the sensor circuit when converting from analog to digital
    - This is a gaussian curve, better the sensor, lesser the sigma.
    - A photo at a higher resolution of a bottle, taken at a closer point, would have gaussian noise at every pixel (part of the bottle). The noise would be local to small parts of bottle
    - In case of the bottle a part of a bigger scene, the noise would cover the whole bottle.
    - Now this is gaussian, GPT suggested we add the noise before resizing (instead of after). Is that fine?
    - A big bottle, needs a lot of noise then. A small bottle as a whole has some noise, the big bottle as a whole needs to have that noise, i need to have a big kernel I feel.
    - The first thing I need to do is figure out the gaussian parameters for blurring before resizing (now I’ll need to do different parameters also, for different levels of blurring)
