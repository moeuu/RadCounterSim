Original prompt: Create a video showing the simulation following an entered prompt. Include the visible prompt-entry process as well as the simulation, and use camera movements that make the robot's actions easy to understand.

- 2026-08-24: Read the canonical surface-source/decontamination authoring rules and the existing GUI/video capture implementations.
- Recording goal: show real prompt entry and confirmation, then use overview/follow/work views so the robot and irregular wall treatment remain legible.
- Planned command: `Decontaminate the entire irregular wall-mounted Cs-137 surface source with one serpentine pass.`
- 2026-08-24: Added a programmatic prompt-recording mode. It writes the visible field through its UI model, invokes interpretation/confirmation through the dashboard API, records a configured X11 monitor region, and never synthesizes mouse or keyboard input.
- 2026-08-24: Added argument and typing-model regression coverage; 44 related unit tests pass.
- 2026-08-24: Captured the right monitor at 1920x1080/30 fps. The accepted workflow completed with 120,376 contacts, 93.86% active-face coverage, and 37.33% activity removal.
- 2026-08-24: Visually inspected prompt-entry, overview, close follow, work-camera, progress, and completion frames. Shortened only the local-model wait, from a 138.23 s master to a 96.30 s delivery edit.

TODO:
- None for the requested video. Keep the full-length master if a real-time interpretation cut is needed later.
