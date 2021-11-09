# Fisheries Size and Functional Type Model (FEISTY)

FEISTY is a mechanistic model that includes multiple fish functional types and size-classes. {cite:t}`Petrik-Stock-etal-2019` introduced FEISTY and its underlying parameterization. This packaged provides a Python interface to FEISTY, using [xarray](http://xarray.pydata.org) and other packages.



```{figure} images/feisty-schematic.png
---
height: 400px
name: feisty-schematic
---
FEISTY model structure showing two zooplankton and three fish size, three functional types, three habitats, two prey categories, and feeding interactions (arrows).
Dashed arrow denotes feeding only occurs in shelf regions with depth < 200 m.
The dotted line surrounds zooplankton biomass that is input from an ESM.
Taken from {cite:t}`Petrik-Stock-etal-2019`.
```
