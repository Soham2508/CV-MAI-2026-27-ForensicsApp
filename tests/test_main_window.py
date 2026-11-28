from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from forensics_app.ui.main_window import MainWindow, touchpad_delta_y


class MainWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = MainWindow.__new__(MainWindow)
        self.window.sidebar_canvas = MagicMock()
        self.window.sidebar_canvas.winfo_rootx.return_value = 10
        self.window.sidebar_canvas.winfo_rooty.return_value = 20
        self.window.sidebar_canvas.winfo_width.return_value = 200
        self.window.sidebar_canvas.winfo_height.return_value = 300
        self.window.sidebar = MagicMock()

    @patch("forensics_app.ui.main_window.sys.platform", "win32")
    def test_mousewheel_scrolls_sidebar_when_pointer_is_over_canvas(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=None, delta=120)

        result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(-1, "units")

    def test_linux_mousewheel_scrolls_down_when_pointer_is_over_canvas(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=5, delta=0)

        result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(1, "units")

    def test_mousewheel_does_not_scroll_sidebar_when_pointer_is_outside(self) -> None:
        event = SimpleNamespace(x_root=500, y_root=100, num=None, delta=120)

        result = self.window._on_sidebar_mousewheel(event)

        self.assertIsNone(result)
        self.window.sidebar_canvas.yview_scroll.assert_not_called()

    def test_mousewheel_scrolls_when_event_widget_is_inside_sidebar(self) -> None:
        child_button = MagicMock()
        child_button.master = self.window.sidebar_canvas
        # Even if x_root and y_root are 0 (common quirk on macOS Aqua)
        event = SimpleNamespace(
            widget=child_button, x_root=0, y_root=0, num=None, delta=120
        )

        result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once()

    def test_macos_mousewheel_scrolls_down_with_native_delta(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=None, delta=-3)

        with patch("sys.platform", "darwin"):
            result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(3, "units")

    def test_macos_mousewheel_scrolls_up_with_native_delta(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=None, delta=2)

        with patch("sys.platform", "darwin"):
            result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(-2, "units")

    def test_macos_mousewheel_fractional_delta_moves_at_least_one_unit(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=None, delta=-0.4)

        with patch("sys.platform", "darwin"):
            result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(1, "units")


class Tk9SidebarScrollTests(unittest.TestCase):
    """Tk 9: wheel deltas of 120 per notch and trackpad <TouchpadScroll> events."""

    def setUp(self) -> None:
        self.window = MainWindow.__new__(MainWindow)
        self.window._tk_major = 9
        self.window.sidebar_canvas = MagicMock()
        self.window.sidebar_canvas.winfo_rootx.return_value = 10
        self.window.sidebar_canvas.winfo_rooty.return_value = 20
        self.window.sidebar_canvas.winfo_width.return_value = 200
        self.window.sidebar_canvas.winfo_height.return_value = 300
        self.window.sidebar = MagicMock()

    def scrolled_units(self) -> int:
        calls = self.window.sidebar_canvas.yview_scroll.call_args_list
        return sum(c.args[0] for c in calls)

    def test_touchpad_delta_decoding(self) -> None:
        self.assertEqual(touchpad_delta_y(5), 5)
        self.assertEqual(touchpad_delta_y(0xFFFF), -1)
        self.assertEqual(touchpad_delta_y((3 << 16) | 0xFFF6), -10)  # x=3, y=-10

    def test_wheel_notch_scrolls_like_tk_widgets(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=None, delta=-120)
        self.assertEqual(self.window._on_sidebar_mousewheel(event), "break")
        # -delta / 4 = 30 pixels = 1 unit of 20 px, 10 px kept for later
        self.assertEqual(self.scrolled_units(), 1)
        self.window._on_sidebar_mousewheel(event)
        self.assertEqual(self.scrolled_units(), 3)

    def test_touchpad_scroll_accumulates_small_moves(self) -> None:
        up = (0 << 16) | (-6 & 0xFFFF)  # finger moves up: content scrolls down
        for _ in range(3):
            event = SimpleNamespace(x_root=100, y_root=100, delta=up)
            self.assertEqual(self.window._on_sidebar_touchpad(event), "break")
        self.assertEqual(self.scrolled_units(), 0)  # 18 px: less than a unit
        self.window._on_sidebar_touchpad(
            SimpleNamespace(x_root=100, y_root=100, delta=up)
        )
        self.assertEqual(self.scrolled_units(), 1)
        down = SimpleNamespace(x_root=100, y_root=100, delta=45)
        self.window._on_sidebar_touchpad(down)
        self.assertEqual(self.scrolled_units(), -1)

    def test_touchpad_outside_sidebar_is_ignored(self) -> None:
        event = SimpleNamespace(x_root=500, y_root=100, delta=-40)
        self.assertIsNone(self.window._on_sidebar_touchpad(event))
        self.window.sidebar_canvas.yview_scroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
