Final goal: generate different blurred versions of a given image. 
My model works for a specific blur level. Images that are close in the road scene get segmented, but those farther away or slightly blurred are hard for the model to capture. 

I want to work on generating different blurred counterparts of the image. First, we take multiple photos of a single object in a scene and see how the blurred objects look in each image. It would be useful to generate blurred versions for different kinds of images in various locations and see how they look in the end, so we can create faithful representations of existing images in our system. 

# Measuring or getting a feel of blur
We want to know how the same object looks in different positions in the same image.