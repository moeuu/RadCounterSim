"""Dedicated Kit extension entry point for the RadInterAct dashboard."""

from __future__ import annotations

import omni.ext

from .ui.dashboard import RadCounterDashboard


class RadCounterDashboardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._dashboard = RadCounterDashboard(ext_id)

    def on_shutdown(self) -> None:
        if self._dashboard is not None:
            self._dashboard.shutdown()
        self._dashboard = None
