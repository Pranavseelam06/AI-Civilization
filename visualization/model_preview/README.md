# Local model preview

From the project root run `python -m visualization.model_preview`, then open
http://127.0.0.1:8000. Stop the server with Ctrl+C.

Drag to rotate, scroll to zoom, right-drag to pan. Choose any of the four
characters from the dropdown. Reset camera restores the starting view.

Models are centered and scaled automatically: the grass tile spans four units,
adults are 1.6 units tall, children 1.1, and the berry bush 0.8. These are preview
sizes, not simulation rules. These assets use Z as vertical and are rotated to
the viewer's Y-up coordinates before sizing. Simulation z will map to Y when
we connect a 3D simulation viewer later. No animations are included in these GLBs.

The files stay on your computer. The viewer downloads the free, pinned Three.js
0.180.0 library from jsDelivr, so opening it requires internet access. There is
no API key, subscription, or paid hosting. This is an asset preview only.
