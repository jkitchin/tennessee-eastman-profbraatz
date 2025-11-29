# Process Variables Reference

## Measurements (XMEAS 1-41)

All measurements are accessed via `result.measurements[:, idx]` where `idx` is 0-based (XMEAS(1) = index 0).

### Continuous Measurements (XMEAS 1-22)

| # | Index | Name | Units | Description |
|---|-------|------|-------|-------------|
| 1 | 0 | A Feed | kscmh | A Feed flow (stream 1) |
| 2 | 1 | D Feed | kg/hr | D Feed flow (stream 2) |
| 3 | 2 | E Feed | kg/hr | E Feed flow (stream 3) |
| 4 | 3 | A and C Feed | kscmh | A and C Feed flow (stream 4) |
| 5 | 4 | Recycle Flow | kscmh | Recycle flow (stream 8) |
| 6 | 5 | Reactor Feed Rate | kscmh | Reactor feed rate (stream 6) |
| 7 | 6 | Reactor Pressure | kPa gauge | **Key safety variable** |
| 8 | 7 | Reactor Level | % | **Key safety variable** |
| 9 | 8 | Reactor Temperature | deg C | **Key safety variable** |
| 10 | 9 | Purge Rate | kscmh | Purge rate (stream 9) |
| 11 | 10 | Product Sep Temp | deg C | Separator temperature |
| 12 | 11 | Product Sep Level | % | **Key safety variable** |
| 13 | 12 | Prod Sep Pressure | kPa gauge | Separator pressure |
| 14 | 13 | Prod Sep Underflow | m³/hr | Separator underflow (stream 10) |
| 15 | 14 | Stripper Level | % | **Key safety variable** |
| 16 | 15 | Stripper Pressure | kPa gauge | Stripper pressure |
| 17 | 16 | Stripper Underflow | m³/hr | Stripper underflow (stream 11) |
| 18 | 17 | Stripper Temperature | deg C | Stripper temperature |
| 19 | 18 | Stripper Steam Flow | kg/hr | Steam flow |
| 20 | 19 | Compressor Work | kW | Compressor power |
| 21 | 20 | Reactor CW Outlet Temp | deg C | Cooling water outlet |
| 22 | 21 | Separator CW Outlet Temp | deg C | Separator CW outlet |

### Sampled Measurements (XMEAS 23-41)

These have sampling delays (6-15 minute dead time).

**Reactor Feed Analysis (XMEAS 23-28)** - Stream 6, mol%
| # | Index | Component |
|---|-------|-----------|
| 23 | 22 | A |
| 24 | 23 | B |
| 25 | 24 | C |
| 26 | 25 | D |
| 27 | 26 | E |
| 28 | 27 | F |

**Purge Gas Analysis (XMEAS 29-36)** - Stream 9, mol%
| # | Index | Component |
|---|-------|-----------|
| 29 | 28 | A |
| 30 | 29 | B |
| 31 | 30 | C |
| 32 | 31 | D |
| 33 | 32 | E |
| 34 | 33 | F |
| 35 | 34 | G |
| 36 | 35 | H |

**Product Analysis (XMEAS 37-41)** - Stream 11, mol%
| # | Index | Component |
|---|-------|-----------|
| 37 | 36 | D |
| 38 | 37 | E |
| 39 | 38 | F |
| 40 | 39 | G |
| 41 | 40 | H |

---

## Manipulated Variables (XMV 1-12)

All MVs are valve positions (0-100%). Access via `result.manipulated_vars[:, idx]` where `idx` is 0-based.

| # | Index | Name | Description |
|---|-------|------|-------------|
| 1 | 0 | D Feed Flow | D feed valve (stream 2) |
| 2 | 1 | E Feed Flow | E feed valve (stream 3) |
| 3 | 2 | A Feed Flow | A feed valve (stream 1) |
| 4 | 3 | A and C Feed Flow | A+C feed valve (stream 4) |
| 5 | 4 | Compressor Recycle Valve | Recycle valve |
| 6 | 5 | Purge Valve | Purge valve (stream 9) |
| 7 | 6 | Separator Pot Liquid Flow | Separator underflow valve |
| 8 | 7 | Stripper Product Flow | Stripper underflow valve |
| 9 | 8 | Stripper Steam Valve | Steam valve |
| 10 | 9 | Reactor Cooling Water | **Critical for IDV(4)** |
| 11 | 10 | Condenser Cooling Water | **Critical for IDV(5)** |
| 12 | 11 | Agitator Speed | Reactor agitator |

### Setting MVs Manually

```python
sim.set_mv(10, 50.0)  # Set reactor cooling water to 50%
```

---

## Disturbances (IDV 1-20)

| IDV | Type | Description | Effect |
|-----|------|-------------|--------|
| 1 | Step | A/C Feed Ratio, B Composition Constant | Feed composition change |
| 2 | Step | B Composition, A/C Ratio Constant | Feed composition change |
| 3 | Step | D Feed Temperature | D feed temp step |
| 4 | Step | Reactor Cooling Water Inlet Temperature | **Common test fault** - CW temp rise |
| 5 | Step | Condenser Cooling Water Inlet Temperature | Condenser CW temp rise |
| 6 | Step | A Feed Loss | **Severe** - A feed drops to zero |
| 7 | Step | C Header Pressure Loss | Reduced C availability |
| 8 | Random | A, B, C Feed Composition | Random composition variation |
| 9 | Random | D Feed Temperature | Random temp variation |
| 10 | Random | C Feed Temperature | Random temp variation |
| 11 | Random | Reactor Cooling Water Inlet Temperature | Random CW variation |
| 12 | Random | Condenser Cooling Water Inlet Temperature | Random CW variation |
| 13 | Drift | Reaction Kinetics | Slow kinetics drift |
| 14 | Sticking | Reactor Cooling Water Valve | Valve sticks |
| 15 | Sticking | Condenser Cooling Water Valve | Valve sticks |
| 16-20 | Unknown | Reserved for testing | Undocumented |

### Activating Disturbances

```python
# During simulation setup
result = sim.simulate(
    duration_hours=4.0,
    disturbances={4: (1.0, 1)}  # IDV(4) at t=1 hour
)

# Or dynamically
sim.set_disturbance(4, 1)   # Turn on
sim.set_disturbance(4, 0)   # Turn off
sim.clear_disturbances()    # Turn off all
```

---

## Safety Limits

Process shutdown occurs if any of these limits are exceeded:

| Variable | Limit | Measurement |
|----------|-------|-------------|
| Reactor Pressure | > 3000 kPa | XMEAS(7) |
| Reactor Level | < 2% or > 24% | XMEAS(8) |
| Reactor Temperature | > 175°C | XMEAS(9) |
| Separator Level | < 1% or > 12% | XMEAS(12) |
| Stripper Level | < 1% or > 8% | XMEAS(15) |

---

## Common Sensor Groups

### Reactor Monitoring
```python
reactor_pressure = result.measurements[:, 6]   # XMEAS(7)
reactor_level = result.measurements[:, 7]      # XMEAS(8)
reactor_temp = result.measurements[:, 8]       # XMEAS(9)
```

### Feed Flows
```python
a_feed = result.measurements[:, 0]     # XMEAS(1)
d_feed = result.measurements[:, 1]     # XMEAS(2)
e_feed = result.measurements[:, 2]     # XMEAS(3)
ac_feed = result.measurements[:, 3]    # XMEAS(4)
```

### Levels
```python
reactor_level = result.measurements[:, 7]     # XMEAS(8)
separator_level = result.measurements[:, 11]  # XMEAS(12)
stripper_level = result.measurements[:, 14]   # XMEAS(15)
```

### Cooling Water Related
```python
reactor_cw_out = result.measurements[:, 20]   # XMEAS(21)
separator_cw_out = result.measurements[:, 21] # XMEAS(22)
reactor_cw_valve = result.manipulated_vars[:, 9]   # XMV(10)
condenser_cw_valve = result.manipulated_vars[:, 10] # XMV(11)
```

---

## Component Names

The 8 chemical components in the process:

| Code | Name | Role |
|------|------|------|
| A | Component A | Reactant (non-condensable) |
| B | Component B | Reactant (non-condensable) |
| C | Component C | Reactant (non-condensable) |
| D | Component D | Reactant (condensable) |
| E | Component E | Reactant (condensable) |
| F | Component F | Intermediate product |
| G | Component G | **Product** |
| H | Component H | **Product** |

Main reactions:
- A(g) → G(liq) + H(liq)
- A(g) + D(g) → F(liq)
- 3D(g) → 2F(liq)
