- I've got a A small model, A medium model. smallnet
- A high recall negmasker for mediujm, a stable for medium, an sm model
- Segmentations


I want to run these on delhi dataset, the dataset needs to be chunked though. Since colab is ephemeral, I would prefer losing at max 20 minutes of work. 256 smallnet in 5 seconds -> 50 images per second in smallnet. 3000 images per minute

since 3 models, 1000 per minute
negmask assume similar speed along with segmask. So assume 300 images per minute. 
300 * 20 = 6000 images per chunk

Once we are done with a chunk. we tar and push it.  We can use the dvctar script for that. Will need some testing, but its okay. We push to the same location.  

