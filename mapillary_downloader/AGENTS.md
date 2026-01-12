# Project structure and code walkthrough


A CLI to Download all street-level images for a city from Mapillary.
The entrypoint is `cli.py`
The code uses SQLite to manage datbase state, the interface is present in `state.py`
`api_client.py` contains code to talk to mapillary. 
In both `state.py` and `api_client.py` the functions should be atomic and testable.

`orchestrator.py` contains the main code for:
- finding the bounding box for a given city using `geocoder.py`
- tiling the box using `tiles.py`
- We split a box in tiles, and split each tile in turn to a list of smaller bounding boxes
- For downloading, we query mapillary for image IDs for each bounding box, get the list and persist them in the database for later downloading. This is the discovery phase.  
- once discovery is done, we query the database to get images that are not yet downloaded and download them.

The CLI should be crash-recoverable always. We should be able to restart downloads fastly

The downloaded data is dumped in the output folder, along with the state (SQLite DB file). State is managed per output folder.

# Project tools

- Use `uv` to manage the project. Running the downloader: `uv run python -m mapillary_downloader --city "Palo Alto, CA" --output ./images`
- Running tests: `uv run pytest`
