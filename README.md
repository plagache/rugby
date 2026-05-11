# rugby
analyzing rugby matches

## install
```sh
rm -rf .venv && uv venv --python 3.12 .venv && uv pip install -e .
```

## run

```sh
uv run yolo_inference.py
```

## Todo

we can train a small Alex net, instead of kmean for better accuracy in detecting Teams
actually we can make a specfique architecture for Teams detection

- [x] change bounding box to label at bottom of object
    - can we add a little box arround the label ?
    - can we add opacity on label ?
- fine tune yolo with the roboflow dataset # need Graphic Card
