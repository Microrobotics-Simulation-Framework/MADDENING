# maddening.viz

Visualization subsystem: relay, runner, renderers, and network transport.

## Architecture

```
GraphManager  -->  StateRelay  -->  Renderer(s)
     |                                  |
 (observer)                      (poll latest_snapshot)
```

The simulation thread writes snapshots via the observer pattern. Renderers poll at their own display rate. This decouples simulation speed from frame rate.

## Core Components

### GraphInfo (`renderer.py`)

Read-only metadata snapshot passed to renderers at setup time.

```python
info = GraphInfo.from_graph_manager(gm)
# info.node_names, info.node_params, info.node_state_fields, info.edges, info.timestep
```

### Renderer ABC (`renderer.py`)

```python
class Renderer(ABC):
    def setup(self, graph_info: GraphInfo) -> None: ...
    def update(self, sim_time: float, state: dict) -> None: ...
    def teardown(self) -> None: ...
    def requested_fields(self) -> dict | None: ...  # optional filter
```

### StateRelay (`relay.py`)

Thread-safe one-slot buffer. Attaches to GraphManager as an observer.

```python
relay = StateRelay()
relay.attach(gm)
sim_time, snapshot = relay.latest_snapshot()  # (float, dict | None)
```

### RealtimeRunner (`runner.py`)

Drives a GraphManager on a background daemon thread, paced to wall-clock time.

```python
runner = RealtimeRunner(gm, relay, time_scale=1.0, command_receiver=None)
runner.start()
runner.pause()
runner.resume()
runner.stop()
runner.time_scale = 2.0   # double speed
runner.sim_time             # current sim time
```

If a `CommandReceiver` is provided, its `latest_commands()` feeds `external_inputs` each step.

## Network Transport (`network.py`)

ZMQ-based remote visualization and command input. All use `CONFLATE` (latest-value-only).

| Class | Role | Default address | Direction |
|-------|------|-----------------|-----------|
| `NetworkRelay` | Publish state (sim side) | `tcp://*:5555` | sim -> viz |
| `NetworkReceiver` | Subscribe to state (viz side) | `tcp://localhost:5555` | sim -> viz |
| `CommandPublisher` | Send commands (controller side) | `tcp://*:5556` | controller -> sim |
| `CommandReceiver` | Receive commands (sim side) | `tcp://localhost:5556` | controller -> sim |

`NetworkReceiver` exposes `latest_snapshot()` with the same interface as `StateRelay`, so renderers work as drop-in replacements.

```python
# Sim side
relay = NetworkRelay("tcp://*:5555")
relay.attach(gm)

# Viz side
receiver = NetworkReceiver("tcp://localhost:5555")
receiver.start()
sim_time, snapshot = receiver.latest_snapshot()
```

## Backends

### Matplotlib (`backends/matplotlib_renderer.py`)

**MatplotlibTimeSeriesRenderer** -- live line plots of state fields.

```python
ts = MatplotlibTimeSeriesRenderer(relay, plot_config={
    "fields": {"ball": ["position", "velocity"]},
    "window": 500,       # visible data points (0 = all)
    "title": "My Plot",
    "figsize": (10, 6),
})
ts.setup(graph_info)
```

**MatplotlibSceneRenderer** -- 2D animated scene with circle, surface, and hline objects.

```python
scene = MatplotlibSceneRenderer(relay, scene_config={
    "xlim": (-1, 1), "ylim": (-1, 6),
    "objects": [
        {"type": "circle", "node": "ball", "y": "position", "radius": 0.2, "color": "red"},
        {"type": "surface", "node": "table", "y": "position", "depth": 0.5, "color": "#8B7355"},
    ],
})
scene.setup(graph_info)
```

**run_matplotlib** -- drive one or more renderers from a single event loop.

```python
run_matplotlib(scene, timeseries, interval_ms=33)  # blocks on plt.show()
```

### Terminal (`backends/terminal_renderer.py`)

Uses `rich` for flicker-free in-place updates. No GUI needed -- works over SSH.

```python
term = TerminalRenderer(relay, config={
    "fields": {"ball": ["position", "velocity"]},
    "precision": 4,
    "refresh_hz": 20,
    "title": "Monitor",
})
term.setup(graph_info)
term.run_event_loop()          # blocks main thread (Ctrl-C to stop)
# OR
term.start_background()        # background thread (use with matplotlib)
```

## Typical Usage

```python
relay = StateRelay()
relay.attach(gm)
info = GraphInfo.from_graph_manager(gm)
renderer = MatplotlibTimeSeriesRenderer(relay, plot_config={...})
renderer.setup(info)
runner = RealtimeRunner(gm, relay)
runner.start()
run_matplotlib(renderer)   # blocks
runner.stop()
```
