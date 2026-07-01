import unittest

from summary_controller import SummaryController


class FakeApp:
    def __init__(self):
        self._pending_shot_sides = {}
        self._manual_shot_sides = {}
        self._manual_recalc_version = 0
        self._selected_shot_id = None
        self.invalidated = False
        self.graph_updates = 0
        self.summary_updates = 0
        self.highlight_updates = 0
        self.rows = [{
            "shot_id": 1,
            "shooting_hand": "right",
            "power_leg": "left",
        }]
        self.auto_sides = ("right", "left")

    def _get_analysis_rows(self):
        return self.rows

    def _get_auto_shot_sides(self, shot_id):
        self.auto_shot_id = shot_id
        return self.auto_sides

    def _invalidate_analysis_cache(self):
        self.invalidated = True

    def update_full_graph(self):
        self.graph_updates += 1

    def update_highlight(self):
        self.highlight_updates += 1


class TestSummaryController(SummaryController):
    def refresh_summary_table(self):
        self.app.summary_updates += 1


class SummaryControllerTest(unittest.TestCase):
    def test_toggle_pending_side_flips_from_current_and_pending_value(self):
        app = FakeApp()
        controller = TestSummaryController(app)

        controller.toggle_pending_shot_side(1, "hand")
        controller.toggle_pending_shot_side(1, "hand")

        self.assertEqual(app._pending_shot_sides[1]["shooting_hand"], "right")
        self.assertEqual(app._selected_shot_id, 1)
        self.assertEqual(app.summary_updates, 2)

    def test_apply_pending_side_stores_manual_override_and_refreshes_views(self):
        app = FakeApp()
        controller = TestSummaryController(app)
        app._pending_shot_sides[1] = {"shooting_hand": "left"}

        controller.apply_pending_shot_sides(1)

        self.assertEqual(
            app._manual_shot_sides[1],
            {"shooting_hand": "left", "power_leg": "left"})
        self.assertNotIn(1, app._pending_shot_sides)
        self.assertEqual(app._manual_recalc_version, 1)
        self.assertTrue(app.invalidated)
        self.assertEqual(app.graph_updates, 1)
        self.assertEqual(app.summary_updates, 1)
        self.assertEqual(app.highlight_updates, 1)

    def test_apply_auto_matching_sides_removes_manual_override(self):
        app = FakeApp()
        controller = TestSummaryController(app)
        app._manual_shot_sides[1] = {"shooting_hand": "left"}
        app._pending_shot_sides[1] = {
            "shooting_hand": "right",
            "power_leg": "left",
        }

        controller.apply_pending_shot_sides(1)

        self.assertNotIn(1, app._manual_shot_sides)

    def test_selecting_current_shot_does_not_redraw(self):
        app = FakeApp()
        app._selected_shot_id = 1
        controller = TestSummaryController(app)

        controller.select_shot(1)

        self.assertEqual(app.graph_updates, 0)
        self.assertEqual(app.summary_updates, 0)
        self.assertEqual(app.highlight_updates, 0)


if __name__ == "__main__":
    unittest.main()
