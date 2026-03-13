# Simple net

Standard resnet but trained on garbage.  

## analyse the first convolution

- See input channels separately
- See output separately

Today, we do a single kernel. The first one. Thats it. The first layer takes in 3 channels, has 64 kernels, each kernel gives 1 channel, outputs 64 channels, one for each kernel.  

The first thing we see is that all the input channels look very similar for input (RGB is very similar). They are very "aligned". This won't be the case of networks in the inner layers.  

- The first image is quite weird, it seems like noise only
