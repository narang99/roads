Types of data preparation


- Creating crops out of label studio stuff
- Creating image folders with their segmentation masks saved
- Creating synthetic data



Directory structure
```
crops
  1
    raw.png
    proc.png
    meta.json
  ...so on

images
  1
    image.png
    seg.pkl
```

crops/meta.json
```json
{
    "original_path": "absolute path in my honme directory",  // only useful in my local machine, not very useful generally
    "bounding_box": {
        "r", "c", "r_len", "c_len",
    },
}
```